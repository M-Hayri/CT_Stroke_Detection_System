"""
model/ai-servisi/app.py -- BT kesitlerinde kanama bolgesini isaretleyen
U-Net cikarim servisi (FastAPI).

Burada hicbir model egitilmez; yalniz onceden egitilmis 5 checkpoint
(unet_focal_tversky_fold0..4.pt) yuklenir ve gelen goruntu icin bu 5 modelin
olasilik tahminlerinin ortalamasi alinir (5-fold topluluk / ensemble).

U-Net mimarisi ve on-isleme fonksiyonlari model/cekirdek/ paketinden
import edilir, burada tekrar yazilmaz:
  - cekirdek.unet_model.UNet
  - cekirdek.eval_unet.load_input_1ch (egitimdekiyle birebir ayni on-isleme:
    oku -> gri -> kafa maskesi -> maske disini sifirla -> 224x224 -> /255.0)
  - cekirdek.confound_check.load_colored_224 / region_stats (girdi gorselinin
    beklenen gri yogunluk araligindan [112,142] sapip sapmadigini kontrol eder)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

# --------------------------------------------------------------------------
# YOL COZUMU
# CTStrokeDetectionSystem/model/ai-servisi/app.py -> CTStrokeDetectionSystem/frontend/wwwroot/uploads/masks
# --------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
WWWROOT_DIR = PROJECT_ROOT / "frontend" / "wwwroot"
MASKS_OUTPUT_DIR = WWWROOT_DIR / "uploads" / "masks"
AGIRLIK_DIR = PROJECT_ROOT / "model" / "agirliklar"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.cekirdek.unet_model import UNet  # noqa: E402
from model.cekirdek.eval_unet import (  # noqa: E402
    OUT_SIZE, OPEN_KERNEL, M3_CLEAN_PX_THR, load_input_1ch,
)
from model.cekirdek.confound_check import load_colored_224, region_stats  # noqa: E402

RESULTS_JSON_PATH = AGIRLIK_DIR / "unet_results.json"
LOG_PATH = AGIRLIK_DIR / "unet_log.txt"
CKPT_DIR = AGIRLIK_DIR

# --------------------------------------------------------------------------
# SABITLER
# --------------------------------------------------------------------------
KOL_ADI = "focal_tversky"        # servis edilen kayip fonksiyonu kolu (egitimde secildi)
FOLD_SAYISI = 5
MODEL_SURUMU = f"unet_{KOL_ADI}_ensemble_v1"
LEZYON_ESIK_PX = 50              # bu pikselden kucuk isaretli alan "inme yok" sayilir
GRAY_BAND = (112.0, 142.0)       # beklenen gri yogunluk medyani araligi (girdi denetimi icin)

SINIRLAMALAR = {
    "kapsam": "Yalniz kanama icin dogrulanmistir. Iskemik lezyonlarda performans olculmemistir.",
    "hacim": "Isaretlenen alan gercek lezyondan sistematik olarak dusuktur (olculen oran ~0.72-0.76).",
    "cokOdak": "Birden fazla odak varsa bir kismi isaretlenmemis olabilir (ikincil odak yakalama 0.63).",
    "yanlisPozitif": "Orta hat ve damar yapilarinda yanlis isaretleme gozlenmistir.",
    "kalibrasyon": "Guven skoru KALIBRE EDILMEMISTIR; klinik olasilik olarak yorumlanamaz.",
    "uyari": "Bu arac tani amacli degildir. Hekim degerlendirmesinin yerine gecmez.",
}

CKPT_PATHS = [CKPT_DIR / f"unet_{KOL_ADI}_fold{i}.pt" for i in range(FOLD_SAYISI)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# LOG (model/agirliklar/unet_log.txt dosyasina EKLENIR, uzerine yazilmaz)
# --------------------------------------------------------------------------
logger = logging.getLogger("btinme_unet_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_fh)
    logger.addHandler(_sh)
    logger.propagate = False


# --------------------------------------------------------------------------
# ACILIS KONTROLU: 5 checkpoint SHA-256'si unet_results.json ->
# j4.j4_0_alignment_health.sha256 ile birebir eslesmezse SERVIS BASLAMAZ.
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _beklenen_sha_sozlugu() -> dict:
    with open(RESULTS_JSON_PATH, encoding="utf-8") as fh:
        results = json.load(fh)
    try:
        return results["j4"]["j4_0_alignment_health"]["sha256"]
    except KeyError as exc:
        raise RuntimeError(
            f"ACILIS KONTROLU BASARISIZ: {RESULTS_JSON_PATH} icinde "
            "j4.j4_0_alignment_health.sha256 anahtari bulunamadi."
        ) from exc


def _checkpointleri_dogrula_ve_yukle():
    beklenen = _beklenen_sha_sozlugu()
    modeller, sha_onekleri, sha_tam = [], [], []
    for i, p in enumerate(CKPT_PATHS):
        if not p.exists():
            raise RuntimeError(f"ACILIS KONTROLU BASARISIZ: checkpoint bulunamadi: {p}")
        gercek = _sha256(p)
        anahtar = f"ckpt_{KOL_ADI}_fold{i}"
        beklenen_deger = beklenen.get(anahtar)
        if beklenen_deger is None:
            raise RuntimeError(
                f"ACILIS KONTROLU BASARISIZ: unet_results.json icinde '{anahtar}' anahtari yok."
            )
        if gercek != beklenen_deger:
            raise RuntimeError(
                f"ACILIS KONTROLU BASARISIZ: SHA-256 uyusmazligi ({p.name})\n"
                f"  dosyadaki SHA = {gercek}\n  beklenen SHA  = {beklenen_deger}\n"
                "Checkpoint bozulmus veya degistirilmis olabilir -- servis baslatilmiyor."
            )
        ckpt = torch.load(p, map_location=DEVICE)
        model = UNet(in_channels=1, out_channels=1, base=32).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        modeller.append(model)
        sha_onekleri.append(gercek[:8])
        sha_tam.append(gercek)
    return modeller, sha_onekleri, sha_tam


logger.info("=" * 78)
logger.info("BTİNME U-Net cikarim servisi baslatiliyor...")
logger.info(f"kol={KOL_ADI}  mod=ensemble_{FOLD_SAYISI}fold")

try:
    MODELS, CHECKPOINT_SHA_PREFIXES, CHECKPOINT_SHA_FULL = _checkpointleri_dogrula_ve_yukle()
except Exception:
    logger.exception("ACILIS KONTROLU BASARISIZ -- servis baslatilamadi")
    raise

logger.info(f"checkpoint SHA-256 dogrulamasi: OK ({FOLD_SAYISI}/{FOLD_SAYISI} eslesti)")
for _i, _sha in enumerate(CHECKPOINT_SHA_FULL):
    logger.info(f"  fold{_i}  {_sha}")
logger.info(f"cihaz: {DEVICE}")

_t0 = time.time()
with torch.no_grad():
    _dummy = torch.zeros((1, 1, OUT_SIZE, OUT_SIZE), dtype=torch.float32, device=DEVICE)
    for _m in MODELS:
        _m(_dummy)
ISINMA_MS = (time.time() - _t0) * 1000
logger.info(f"isinma tamamlandi: {ISINMA_MS:.1f} ms")

_t1 = time.time()
with torch.no_grad():
    for _m in MODELS:
        _m(_dummy)
TEK_GORUNTU_MS = (time.time() - _t1) * 1000
logger.info(f"tek-goruntu cikarim suresi (isinma sonrasi, {FOLD_SAYISI}-fold toplam): {TEK_GORUNTU_MS:.1f} ms")
logger.info("=" * 78)


# --------------------------------------------------------------------------
# GIRDI DENETIMI
# --------------------------------------------------------------------------
def _girdi_kontrolu(bgr: np.ndarray) -> tuple[dict, np.ndarray]:
    """Yuklenen goruntunun beklenen BT kesiti formatina uyup uymadigini kontrol
    eder (load_colored_224 + region_stats, cekirdek/confound_check.py).
    Donus: (girdiKontrol dict, 224x224 kafa-maskeli renkli resize -- overlay
    arka plani icin yeniden kullanilir)."""
    resized_bgr, head224 = load_colored_224(bgr)
    mask_bool = head224 > 0
    stats = region_stats(resized_bgr, mask_bool)

    sebepler: list[str] = []
    medyan = None
    if stats is None:
        sebepler.append("kafa maskesi bulunamadi (goruntu bos/duz olabilir)")
    else:
        medyan = round(stats["gray_median"], 1)
        if medyan < GRAY_BAND[0] or medyan > GRAY_BAND[1]:
            sebepler.append(
                f"medyan gri yogunluk bant disi: {medyan} "
                f"(beklenen [{GRAY_BAND[0]:.0f},{GRAY_BAND[1]:.0f}])"
            )
        if stats["sat_mean"] > 0.0:
            sebepler.append("renkli girdi (overlay olabilir)")

    h, w = bgr.shape[:2]
    if (w, h) != (OUT_SIZE, OUT_SIZE):
        sebepler.append(
            f"girdi boyutu {w}x{h}, {OUT_SIZE}x{OUT_SIZE} bekleniyor (yeniden boyutlandirildi)"
        )

    girdi_kontrol = {
        "durum": "warn" if sebepler else "ok",
        "medyanYogunluk": medyan,
        "sebep": "; ".join(sebepler) if sebepler else None,
    }
    return girdi_kontrol, resized_bgr


def _yaz_png(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite Windows'ta Turkce/Unicode yollarda (ör. 'BTİNME', 'İ' harfi)
    SESSIZCE basarisiz olur (False doner, exception atmaz) -- tipki read_bgr'nin
    cv2.imread yerine np.fromfile+cv2.imdecode kullanmasinin nedeni gibi, yazma
    tarafinda da ayni sorunu cv2.imencode + duz Python dosya yazimiyla asariz."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG kodlanamadi: {path}")
    path.write_bytes(buf.tobytes())


def _sonislem_uygula(binary_mask: np.ndarray) -> np.ndarray:
    """Opsiyonel: acma (3x3 eliptik, eval_unet.OPEN_KERNEL) + min-alan filtresi
    (eval_unet.M3_CLEAN_PX_THR = 50px). Varsayilan KAPALI (yalniz sonislem=True
    istenirse cagirilir)."""
    mask255 = (binary_mask * 255).astype(np.uint8)
    opened = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, OPEN_KERNEL)
    n_comp, labels, stats_cc, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    kept = np.zeros_like(opened)
    for lbl in range(1, n_comp):
        if stats_cc[lbl, cv2.CC_STAT_AREA] >= M3_CLEAN_PX_THR:
            kept[labels == lbl] = 1
    return kept.astype(np.uint8)


# --------------------------------------------------------------------------
# CIKARIM HATTI (asamalarin sirasi degistirilemez -- egitimdekiyle ayni olmali)
# --------------------------------------------------------------------------
def _cikarim_calistir(bgr: np.ndarray, orig_filename: str | None, sonislem: bool) -> dict:
    girdi_kontrol, resized_bgr224 = _girdi_kontrolu(bgr)

    arr01 = load_input_1ch(bgr)  # egitimdeki load_input_1ch ile BIREBIR ayni zincir
    tensor = torch.from_numpy(arr01).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob_stack = np.stack(
            [torch.sigmoid(m(tensor)).squeeze(0).squeeze(0).cpu().numpy() for m in MODELS],
            axis=0,
        )
    prob_map = prob_stack.mean(axis=0).astype(np.float32)  # 5-fold olasilik ORTALAMASI

    binary_mask = (prob_map > 0.5).astype(np.uint8)
    if sonislem:
        binary_mask = _sonislem_uygula(binary_mask)

    mask255 = (binary_mask * 255).astype(np.uint8)
    n_comp, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask255, connectivity=8)
    odak_alanlari = (
        sorted((int(a) for a in stats_cc[1:, cv2.CC_STAT_AREA]), reverse=True) if n_comp > 1 else []
    )
    odak_sayisi = len(odak_alanlari)
    lezyon_alan_px = int(sum(odak_alanlari))
    inme_var_mi = lezyon_alan_px >= LEZYON_ESIK_PX

    mask_bool = binary_mask.astype(bool)
    guven = float(prob_map[mask_bool].mean()) if mask_bool.any() else 0.0
    maksimum_olasilik = float(prob_map.max())

    # overlay: gri arka plan (modelin gordugu 224x224 kafa-maskeli goruntu) + macenta kontur, dolgu YOK
    gray_bg = cv2.cvtColor(resized_bgr224, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)
    if odak_sayisi > 0:
        contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 255), thickness=2)

    MASKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(orig_filename or "kesit").stem or "kesit"
    mask_file_name = f"mask_{stem}.png"
    overlay_file_name = f"overlay_{stem}.png"
    _yaz_png(MASKS_OUTPUT_DIR / mask_file_name, mask255)
    _yaz_png(MASKS_OUTPUT_DIR / overlay_file_name, overlay)

    if inme_var_mi:
        teshis_detay = f"Kanama supheli bolge isaretlendi (alan {lezyon_alan_px} px, {odak_sayisi} odak)."
    else:
        teshis_detay = "Bu kesitte esik uzerinde (>=50px) bir bulgu isaretlenmedi."

    return {
        "inmeVarMi": bool(inme_var_mi),
        "guvenSkoru": round(guven, 4),
        "maskeYolu": f"/uploads/masks/{mask_file_name}",
        "teshisDetay": teshis_detay,
        "overlayYolu": f"/uploads/masks/{overlay_file_name}",
        "lezyonAlanPx": lezyon_alan_px,
        "odakSayisi": odak_sayisi,
        "odakAlanlariPx": odak_alanlari,
        "maskeIciOrtalamaOlasilik": round(guven, 4),
        "maksimumOlasilik": round(maksimum_olasilik, 4),
        "modelSurumu": MODEL_SURUMU,
        "checkpointSha": CHECKPOINT_SHA_PREFIXES,
        "sonIslemUygulandi": bool(sonislem),
        "girdiKontrol": girdi_kontrol,
        "sinirlamalar": SINIRLAMALAR,
    }


# --------------------------------------------------------------------------
# UC NOKTA: POST /api/stroke-segmentation
# --------------------------------------------------------------------------
app = FastAPI(title="BTİNME U-Net Cikarim Servisi")


@app.post("/api/stroke-segmentation")
async def stroke_segmentation(file: UploadFile = File(...), sonislem: bool = Form(False)):
    req_t0 = time.time()

    ham = await file.read()
    if not ham:
        raise HTTPException(status_code=400, detail="Bos dosya yuklendi.")

    nparr = np.frombuffer(ham, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(
            status_code=400,
            detail=f"Goruntu cozumlenemedi (bozuk veya desteklenmeyen dosya): {file.filename}",
        )

    try:
        payload = _cikarim_calistir(bgr, file.filename, sonislem)
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"cikarim sirasinda beklenmeyen hata (dosya={file.filename})")
        raise HTTPException(
            status_code=500,
            detail="Model calisirken beklenmeyen bir hata olustu; sunucu logunda ayrinti var.",
        )

    sure_ms = (time.time() - req_t0) * 1000
    logger.info(
        f"istek tamamlandi: dosya={file.filename} sure={sure_ms:.1f}ms "
        f"lezyonAlanPx={payload['lezyonAlanPx']} girdiKontrol={payload['girdiKontrol']['durum']}"
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
