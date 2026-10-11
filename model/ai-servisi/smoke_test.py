"""
model/ai-servisi/smoke_test.py -- U-Net cikarim servisi (app.py) icin duman testi.

ONKOSUL: servis zaten ayakta olmali --
    uvicorn app:app --host 127.0.0.1 --port 8000   (model/ai-servisi/ klasorunden)
Bu script servisi baslatmaz/durdurmaz, yalniz HTTP istekleriyle test eder.

NOT: testler 2-4 orijinal TEKNOFEST goruntu setine ve egitim sirasinda uretilen
unet_oof_focal_tversky.npz dosyasina ihtiyac duyar; bu dosya boyut nedeniyle bu
depoya dahil edilmemistir. Bu script bu haliyle yalniz orijinal gelistirme
ortaminda calisir -- burada kod/yontem referansi olarak tutulmaktadir.

Testler:
  1) checkpoint SHA-256 -- app.py'nin acilis kontrolunden BAGIMSIZ bir dogrulama
     (ayni referansa, unet_results.json -> j4.j4_0_alignment_health.sha256'ya karsi)
  2) 5 bilinen pozitif OOF goruntusu (unet_oof_focal_tversky.npz'de en buyuk
     tahmin alanina sahip 5 kayit) -> lezyonAlanPx 0 degil, goruntu alanini asmiyor
  3) 1 bilinen negatif goruntu (OOF tahmin alani=0) -> inmeVarMi=false bekleniyor
  4) kasitli OVERLAY (ham, renkli) goruntu -> girdiKontrol.durum=="warn" bekleniyor
  5) bozuk dosya -> 400 bekleniyor
  6) uctan uca sure olculur

Salt-cikarim: hicbir model egitilmez, hicbir esik degistirilmez. Sonuclar
unet_results.json icine YALNIZ yeni "n1" anahtari olarak yazilir (mevcut
anahtarlara dokunulmaz). Log unet_log.txt'ye EKLENIR.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AGIRLIK_DIR = PROJECT_ROOT / "model" / "agirliklar"
BACKEND_ROOT = PROJECT_ROOT / "backend"
PYTON_DIR = Path(__file__).resolve().parent

RESULTS_JSON_PATH = AGIRLIK_DIR / "unet_results.json"
LOG_PATH = AGIRLIK_DIR / "unet_log.txt"
MANIFEST_PATH = PROJECT_ROOT / "model" / "egitim" / "dataset_manifest.json"
NPZ_PATH = AGIRLIK_DIR / "unet_oof_focal_tversky.npz"  # bu depoda YOK, bkz. ust not
CKPT_DIR = AGIRLIK_DIR

API_BASE = "http://127.0.0.1:8000"
API_URL = f"{API_BASE}/api/stroke-segmentation"
KOL_ADI = "focal_tversky"
FOLD_SAYISI = 5
OUT_SIZE = 224


def tf(v):
    """True/False/None degerini kisa harfe cevirir (log satirlarini kisaltmak icin)."""
    return "NA" if v is None else ("t" if v else "f")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha() -> tuple[bool, dict]:
    results = json.load(open(RESULTS_JSON_PATH, encoding="utf-8"))
    beklenen = results["j4"]["j4_0_alignment_health"]["sha256"]
    detay = {}
    tumu_eslesti = True
    for i in range(FOLD_SAYISI):
        p = CKPT_DIR / f"unet_{KOL_ADI}_fold{i}.pt"
        gercek = _sha256(p)
        beklenen_deger = beklenen.get(f"ckpt_{KOL_ADI}_fold{i}")
        eslesti = gercek == beklenen_deger
        tumu_eslesti = tumu_eslesti and eslesti
        detay[f"fold{i}"] = {"sha256": gercek, "beklenen": beklenen_deger, "eslesti": eslesti}
    return tumu_eslesti, detay


def secim_yap() -> dict:
    """5 bilinen pozitif + 1 bilinen negatif + 1 OVERLAY test goruntusunu
    npz/manifest'ten DETERMINISTIK olarak secer (unsigned-tip tuzagina
    dusmemek icin alan toplami int64'e cast edilir)."""
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    by_id = {it["img_id"]: it for it in manifest}
    npz = np.load(NPZ_PATH, allow_pickle=True)
    areas = npz["pred"].astype(np.int64).reshape(len(npz["pred"]), -1).sum(axis=1)
    is_pos = npz["is_positive"]
    slice_id = npz["slice_id"]

    pos_idx = np.where(is_pos)[0]
    pos_order = pos_idx[np.argsort(areas[pos_idx])[::-1]]
    pozitif_idler = [str(slice_id[i]) for i in pos_order[:5]]

    neg_idx = np.where(~is_pos)[0]
    neg_zero = [i for i in neg_idx if areas[i] == 0]
    negatif_id = (
        str(slice_id[neg_zero[0]]) if neg_zero
        else str(slice_id[neg_idx[np.argmin(areas[neg_idx])]])
    )

    overlay_id, overlay_path = None, None
    for sid in pozitif_idler:
        img_path = Path(by_id[sid]["img_path"])
        aday = Path(str(img_path).replace("\\PNG\\", "\\OVERLAY\\"))
        if aday.exists():
            overlay_id, overlay_path = sid, aday
            break

    return {
        "pozitifler": [(sid, Path(by_id[sid]["img_path"])) for sid in pozitif_idler],
        "negatif": (negatif_id, Path(by_id[negatif_id]["img_path"])),
        "overlay": (overlay_id, overlay_path),
    }


def gonder(dosya_yolu: Path | None, ham_bytes: bytes | None = None, dosya_adi: str = "test.png"):
    t0 = time.time()
    fh = None
    if ham_bytes is not None:
        files = {"file": (dosya_adi, ham_bytes, "application/octet-stream")}
    else:
        fh = open(dosya_yolu, "rb")
        files = {"file": (dosya_yolu.name, fh, "image/png")}
    try:
        resp = requests.post(API_URL, files=files, timeout=60)
    finally:
        if fh is not None:
            fh.close()
    sure_ms = (time.time() - t0) * 1000
    return resp, sure_ms


def main() -> int:
    log_fh = open(LOG_PATH, "a", encoding="utf-8")

    def log(msg: str = ""):
        print(msg)
        log_fh.write(str(msg) + "\n")

    log("=" * 78)
    log(f"duman testi baslatiliyor ({datetime.now().isoformat(timespec='seconds')})")

    try:
        requests.get(API_BASE, timeout=5)
    except requests.exceptions.ConnectionError:
        log(f"HATA: servise ulasilamiyor ({API_URL}).")
        log("Once servisi baslatin (Pyton/ klasorunden):")
        log("  uvicorn app:app --host 127.0.0.1 --port 8000")
        log_fh.close()
        return 1

    # 1) SHA-256 -- bagimsiz dogrulama
    sha_ok, sha_detay = check_sha()
    log(f"1) checkpoint SHA-256 (bagimsiz dogrulama): {'OK' if sha_ok else 'BASARISIZ'}")
    for fold_key, d in sha_detay.items():
        log(f"   {fold_key}: {'eslesti' if d['eslesti'] else 'UYUSMUYOR'}")

    secimler = secim_yap()

    # 2) 5 bilinen pozitif OOF goruntu
    oof_sonuclari = []
    pozitif_sureler = []
    for sid, path in secimler["pozitifler"]:
        resp, sure_ms = gonder(path)
        pozitif_sureler.append(sure_ms)
        alan, sebep = None, None
        ok = resp.status_code == 200
        if ok:
            body = resp.json()
            alan = body["lezyonAlanPx"]
            sebep = body["girdiKontrol"]["sebep"] or ""
            ok = 0 < alan <= OUT_SIZE * OUT_SIZE
        oof_sonuclari.append({"img_id": sid, "status": resp.status_code, "lezyonAlanPx": alan,
                               "sebep": sebep, "passed": ok})
        log(f"   [pozitif {sid}] status={resp.status_code} lezyonAlanPx={alan} "
            f"sure={sure_ms:.1f}ms {'OK' if ok else 'BASARISIZ'}")
    oof_all_passed = all(r["passed"] for r in oof_sonuclari)
    log(f"2) 5 bilinen pozitif OOF goruntu: {'OK' if oof_all_passed else 'BASARISIZ'}")

    # 3) 1 negatif goruntu
    neg_id, neg_path = secimler["negatif"]
    neg_resp, neg_ms = gonder(neg_path)
    neg_inme = None
    neg_ok = neg_resp.status_code == 200
    if neg_ok:
        neg_inme = neg_resp.json()["inmeVarMi"]
        neg_ok = neg_inme is False
    log(f"3) negatif goruntu ({neg_id}): status={neg_resp.status_code} inmeVarMi={neg_inme} "
        f"sure={neg_ms:.1f}ms {'OK' if neg_ok else 'BASARISIZ'}")

    # 4) kasitli OVERLAY (renkli) goruntu
    # NOT: egitim goruntuleri 224x224 degil (cogunlukla 512x512), bu yuzden
    # girdiKontrol.durum HER goruntude (temiz olanlar dahil) "boyut farkli"
    # notuyla zaten "warn" olur -- durum=="warn" TEK BASINA renk-tespitini
    # KANITLAMAZ. Asil kanit: sebep metninde "renkli" gecmesi (doygunluk>0
    # tespiti) VE temiz pozitif orneklerde bu ifadenin GECMEMESI (negatif kontrol).
    overlay_id, overlay_path = secimler["overlay"]
    overlay_ok = False
    overlay_durum = None
    overlay_sebep = None
    overlay_status = None
    overlay_ms = 0.0
    if overlay_path is None:
        log("4) kasitli OVERLAY testi: ATLANDI (uygun OVERLAY dosyasi bulunamadi)")
    else:
        overlay_resp, overlay_ms = gonder(overlay_path)
        overlay_status = overlay_resp.status_code
        if overlay_status == 200:
            gk = overlay_resp.json()["girdiKontrol"]
            overlay_durum = gk["durum"]
            overlay_sebep = gk["sebep"] or ""
            overlay_ok = overlay_durum == "warn" and "renkli" in overlay_sebep
        log(f"4) kasitli OVERLAY goruntu ({overlay_id}): status={overlay_status} "
            f"girdiKontrol.durum={overlay_durum} sebep={overlay_sebep!r} sure={overlay_ms:.1f}ms "
            f"{'OK' if overlay_ok else 'BASARISIZ'}")

    # 4b) negatif kontrol: temiz pozitif orneklerin sebebinde "renkli" HIC GECMEMELI
    # (gecerse doygunluk tespiti yanlis-pozitif veriyor demektir)
    temiz_renk_negatif_kontrol = all("renkli" not in (r["sebep"] or "") for r in oof_sonuclari)
    log(f"4b) negatif kontrol (temiz goruntulerde 'renkli' tespiti YOK): "
        f"{'OK' if temiz_renk_negatif_kontrol else 'BASARISIZ'}")

    # 5) bozuk dosya
    bozuk_bytes = b"BOZUK-DOSYA-ICERIGI-BT-KESITI-DEGIL-0123456789"
    bozuk_resp, bozuk_ms = gonder(None, ham_bytes=bozuk_bytes, dosya_adi="bozuk.png")
    bozuk_ok = bozuk_resp.status_code == 400
    log(f"5) bozuk dosya: status={bozuk_resp.status_code} sure={bozuk_ms:.1f}ms "
        f"{'OK' if bozuk_ok else 'BASARISIZ'}")

    # 6) uctan uca sure
    tum_sureler = pozitif_sureler + [neg_ms] + ([overlay_ms] if overlay_path is not None else [])
    infer_ms_median = float(np.median(tum_sureler)) if tum_sureler else float("nan")
    log(f"6) uctan uca sure: medyan={infer_ms_median:.1f}ms (n={len(tum_sureler)} istek)")

    # 7) eski sahte-sonuc uretme kodunun geri gelmedigini dogrula
    # (GoruntuController.cs'te sabit GuvenSkoru=0.9450 yazan, gercek modele hic
    # baglanmayan bir action vardi; silindi -- bkz. GoruntuController.cs ust yorumu)
    goruntu_cs_text = (BACKEND_ROOT / "Controllers" / "GoruntuController.cs").read_text(encoding="utf-8")
    app_py_text = (PYTON_DIR / "app.py").read_text(encoding="utf-8")
    fake_path_removed = (
        "SegmentasyonYap(int goruntuId)" not in goruntu_cs_text
        and "strokenext_best_model.pt" not in app_py_text
    )
    log(f"7) sahte-sonuc kodu kontrolu (SegmentasyonYap silindi, "
        f"strokenext_best_model.pt referansi yok): {'OK' if fake_path_removed else 'BASARISIZ'}")

    tum_testler_gecti = (
        sha_ok and oof_all_passed and neg_ok and overlay_ok
        and temiz_renk_negatif_kontrol and bozuk_ok
    )

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "bilinmiyor"

    n1 = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arm": KOL_ADI,
        "mode": f"ensemble_{FOLD_SAYISI}fold",
        "sha_match": sha_ok,
        "checkpoint_sha256": {k: v["sha256"] for k, v in sha_detay.items()},
        "device": device,
        "infer_ms_median": infer_ms_median,
        "infer_ms_per_request": tum_sureler,
        "smoke_test": {
            "oof_positive_images": oof_sonuclari,
            "negative_image": {"img_id": neg_id, "status": neg_resp.status_code,
                                "inmeVarMi": neg_inme, "passed": neg_ok},
            "overlay_warn_check": {"img_id": overlay_id, "status": overlay_status,
                                    "durum": overlay_durum, "sebep": overlay_sebep, "passed": overlay_ok},
            "clean_image_color_negative_control": {"passed": temiz_renk_negatif_kontrol},
            "corrupt_file_check": {"status_code": bozuk_resp.status_code, "passed": bozuk_ok},
            "all_passed": tum_testler_gecti,
        },
        "fake_path_removed": fake_path_removed,
    }

    results = {}
    if RESULTS_JSON_PATH.exists():
        results = json.load(open(RESULTS_JSON_PATH, encoding="utf-8"))
    results["n1"] = n1
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as out_fh:
        json.dump(results, out_fh, indent=2, ensure_ascii=False)
    log(f"yazildi: {RESULTS_JSON_PATH} -> 'n1' anahtari")

    log("")
    log(f"kol={KOL_ADI}  mod=ensemble_{FOLD_SAYISI}fold")
    log(f"sha_eslesti={tf(sha_ok)}")
    log(f"cihaz={device}  medyan_cikarim_ms={infer_ms_median:.1f}")
    log(f"duman_testi={'passed' if tum_testler_gecti else 'failed'}")
    log(f"sahte_sonuc_kodu_yok={tf(fake_path_removed)}")
    log("=" * 78)

    log_fh.close()
    return 0 if tum_testler_gecti else 1


if __name__ == "__main__":
    sys.exit(main())
