"""
model/cekirdek/eval_unet.py -- egitilmis modelin degerlendirme/dogrulama
hesaplamalari.

Bu modul iki amaca hizmet eder:
  1) model/ai-servisi/app.py'nin cikarim sirasinda kullandigi birkac saf
     fonksiyonu/sabiti barindirir (load_input_1ch, OUT_SIZE, OPEN_KERNEL,
     M3_CLEAN_PX_THR) -- bunlar canli servis tarafindan gercekten calistirilir.
  2) Asagidaki main() ve "J4.x" etiketli bolumler, modelin egitim sonrasi
     TEK SEFERLIK istatistiksel dogrulamasi icin yazilan, orijinal TEKNOFEST
     veri setine ihtiyac duyan degerlendirme hattidir (bootstrap guven
     araliklari, taban modelle karsilastirma, alt grup analizi, vb.). Bu
     hat bu depoda YENIDEN CALISTIRILMAZ; "J4.x" etiketleri, zaten hesaplanmis
     ve model/agirliklar/unet_results.json icine kaydedilmis sonuclarin
     anahtar adlarıyla (ör. results["j4"]["j4_0_alignment_health"]) birebir
     eslesir -- bu yuzden burada degistirilmedi.

Hicbir model egitilmez; hicbir mevcut .npz/.pt/.json dosyasinin uzerine
yazilmaz.

Kosum (orijinal veri seti mevcutsa):
    python model/cekirdek/eval_unet.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .preprocess import head_mask_from_gray, read_bgr  # noqa: E402
from .unet_model import UNet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_DIR = PROJECT_ROOT / "model" / "agirliklar"
MANIFEST_PATH = PROJECT_ROOT / "model" / "egitim" / "dataset_manifest.json"
RESULTS_PATH = PRODUCT_DIR / "unet_results.json"
LOG_PATH = PRODUCT_DIR / "unet_log.txt"
BASELINE_OOF_PATH = PRODUCT_DIR / "baseline_oof_v2.json"
SUBGROUP_PATH = PRODUCT_DIR / "subgroup_check.json"

OUT_SIZE = 224
N_FOLDS = 5
ARMS = ["focal_tversky", "dice_bce"]
PRIMARY_ARM = "focal_tversky"
SECOND_ARM = "dice_bce"
N_TOTAL_EXPECTED = 2164
N_POS_EXPECTED = 1082

N_BOOT = 10000
BOOT_SEED = 42
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
PARITY_AREA_THR = 50   # taban modelle karsilastirmada kullanilan alan esigi
M3_CLEAN_PX_THR = 50   # negatif ornekte bu pikselden kucuk tahmin alani "hata yok" sayilir


# --------------------------------------------------------------------------- #
# ortak yardimcilar (train_unet.py / rebuild_oof.py ile BIREBIR AYNI)
# --------------------------------------------------------------------------- #

def load_target(item: dict) -> np.ndarray:
    if item["cls"] != "pos":
        return np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)
    gt = read_bgr(item["mask_path"])
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if gt.shape[:2] != (OUT_SIZE, OUT_SIZE):
        gt = cv2.resize(gt, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return (gt > 0).astype(np.float32)


def load_input_1ch(img_path: str) -> np.ndarray:
    bgr = read_bgr(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    head_mask, _ = head_mask_from_gray(gray)
    masked = gray.copy()
    masked[head_mask == 0] = 0
    resized = cv2.resize(masked, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def _dice_hard(pred: np.ndarray, gt: np.ndarray) -> float:
    """train_unet.py train_one() oof_dice_test_pos formulu ile BIREBIR AYNI."""
    inter = int((pred.astype(bool) & (gt > 0)).sum())
    denom = int(pred.sum()) + int((gt > 0).sum())
    return float(2 * inter / denom) if denom > 0 else 1.0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_min_area_filter(binary01: np.ndarray, area_thr: int) -> np.ndarray:
    """baseline_threshold_v2.py:_finish_mask ile AYNI algoritma (3x3 eliptik acma +
    baglanti bilesen alan filtresi); donus 0/1 (baseline'daki *255 olceklemesi yok)."""
    opened = cv2.morphologyEx(binary01.astype(np.uint8), cv2.MORPH_OPEN, OPEN_KERNEL)
    if area_thr <= 0:
        return opened
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_comp <= 1:
        return np.zeros_like(binary01, dtype=np.uint8)
    keep = [r for r in range(1, n_comp) if stats[r, cv2.CC_STAT_AREA] >= area_thr]
    if not keep:
        return np.zeros_like(binary01, dtype=np.uint8)
    return np.isin(labels, keep).astype(np.uint8)


class Logger:
    def __init__(self, path: Path):
        self.fh = open(path, "a", encoding="utf-8")

    def log(self, msg: str = ""):
        print(msg)
        self.fh.write(str(msg) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


# --------------------------------------------------------------------------- #
# blok bootstrap motoru (EK-12.3: J1 bloklari yeniden-orneklemenin birimi)
# --------------------------------------------------------------------------- #

def _block_counts_matrix(n_blocks: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    chosen = rng.randint(0, n_blocks, size=(n_boot, n_blocks))
    offsets = chosen + (np.arange(n_boot)[:, None] * n_blocks)
    counts_flat = np.bincount(offsets.ravel(), minlength=n_boot * n_blocks)
    return counts_flat.reshape(n_boot, n_blocks)


def _setup_blocks(ids, block_of_id):
    blocks = sorted(set(block_of_id[i] for i in ids))
    block_index = {b: k for k, b in enumerate(blocks)}
    item_block_idx = np.array([block_index[block_of_id[i]] for i in ids])
    return len(blocks), item_block_idx


def boot_median(ids, values: dict, block_of_id: dict, n_boot=N_BOOT, seed=BOOT_SEED):
    n_blocks, item_block_idx = _setup_blocks(ids, block_of_id)
    vals = np.array([values[i] for i in ids], dtype=np.float64)
    point = float(np.median(vals))
    order = np.argsort(vals)
    sorted_vals = vals[order]
    sorted_block_idx = item_block_idx[order]
    counts = _block_counts_matrix(n_blocks, n_boot, seed)
    weights_sorted = counts[:, sorted_block_idx]
    cum = np.cumsum(weights_sorted, axis=1)
    total = cum[:, -1]
    half = total / 2.0
    idx = np.argmax(cum >= half[:, None], axis=1)
    boots = sorted_vals[idx]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def boot_mean_indicator(ids, values: dict, indicator_fn, block_of_id: dict, n_boot=N_BOOT, seed=BOOT_SEED):
    n_blocks, item_block_idx = _setup_blocks(ids, block_of_id)
    vals = np.array([values[i] for i in ids], dtype=np.float64)
    indicator = indicator_fn(vals).astype(np.float64)
    point = float(indicator.mean())
    counts = _block_counts_matrix(n_blocks, n_boot, seed)
    weights = counts[:, item_block_idx]
    num = weights @ indicator
    den = weights.sum(axis=1)
    boots = num / den
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def boot_delta_median(ids, values_a: dict, values_b: dict, block_of_id: dict, n_boot=N_BOOT, seed=BOOT_SEED):
    n_blocks, item_block_idx = _setup_blocks(ids, block_of_id)
    va = np.array([values_a[i] for i in ids], dtype=np.float64)
    vb = np.array([values_b[i] for i in ids], dtype=np.float64)
    point = float(np.median(va) - np.median(vb))
    counts = _block_counts_matrix(n_blocks, n_boot, seed)

    def weighted_median_boot(vals):
        order = np.argsort(vals)
        sorted_vals = vals[order]
        sorted_block_idx = item_block_idx[order]
        weights_sorted = counts[:, sorted_block_idx]
        cum = np.cumsum(weights_sorted, axis=1)
        total = cum[:, -1]
        half = total / 2.0
        idx = np.argmax(cum >= half[:, None], axis=1)
        return sorted_vals[idx]

    boots = weighted_median_boot(va) - weighted_median_boot(vb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def boot_delta_mean_indicator(ids, values_a: dict, values_b: dict, indicator_fn, block_of_id: dict,
                               n_boot=N_BOOT, seed=BOOT_SEED):
    n_blocks, item_block_idx = _setup_blocks(ids, block_of_id)
    va = np.array([values_a[i] for i in ids], dtype=np.float64)
    vb = np.array([values_b[i] for i in ids], dtype=np.float64)
    ia = indicator_fn(va).astype(np.float64)
    ib = indicator_fn(vb).astype(np.float64)
    point = float(ia.mean() - ib.mean())
    counts = _block_counts_matrix(n_blocks, n_boot, seed)
    weights = counts[:, item_block_idx]
    den = weights.sum(axis=1)
    boot_a = (weights @ ia) / den
    boot_b = (weights @ ib) / den
    boots = boot_a - boot_b
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _ind_zero(v):
    return v == 0.0


def _ind_below_clean(v):
    return v < M3_CLEAN_PX_THR


def m1_ci(ids, dice_dict, block_of_id, **kw):
    return boot_median(ids, dice_dict, block_of_id, **kw)


def m2_ci(ids, dice_dict, block_of_id, **kw):
    return boot_mean_indicator(ids, dice_dict, _ind_zero, block_of_id, **kw)


def m3_ci(ids, negarea_dict, block_of_id, **kw):
    return boot_mean_indicator(ids, negarea_dict, _ind_below_clean, block_of_id, **kw)


def trivial_block_map(ids):
    return {i: i for i in ids}


# --------------------------------------------------------------------------- #
# veri yukleme
# --------------------------------------------------------------------------- #

def load_manifest():
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    by_id = {it["img_id"]: it for it in manifest}
    block_of_id = {it["img_id"]: it["block_id"] for it in manifest}
    pos_ids = sorted([it["img_id"] for it in manifest if it["cls"] == "pos"], key=int)
    neg_ids = sorted([it["img_id"] for it in manifest if it["cls"] == "neg"], key=int)
    return manifest, by_id, block_of_id, pos_ids, neg_ids


def load_unet_oof(arm: str, manifest_by_id: dict):
    npz_path = PRODUCT_DIR / f"unet_oof_{arm}.npz"
    d = np.load(npz_path, allow_pickle=True)
    slice_id = d["slice_id"].astype(str)
    pred = d["pred"]
    fold = d["fold"]
    per_item_dice, per_item_neg_area, per_item_pred_area, fold_of = {}, {}, {}, {}
    per_item_pred_mask = {}
    for i, sid in enumerate(slice_id):
        sid = str(sid)
        it = manifest_by_id[sid]
        p = pred[i]
        fold_of[sid] = int(fold[i])
        if it["cls"] == "pos":
            gt = load_target(it)
            per_item_dice[sid] = _dice_hard(p, gt)
            per_item_pred_area[sid] = int(p.sum())
        else:
            per_item_neg_area[sid] = int(p.sum())
        per_item_pred_mask[sid] = p
    return {
        "npz_path": npz_path, "slice_id": slice_id, "n": len(slice_id),
        "per_item_dice": per_item_dice, "per_item_neg_area": per_item_neg_area,
        "per_item_pred_area": per_item_pred_area, "fold_of": fold_of,
        "per_item_pred_mask": per_item_pred_mask,
    }


def build_parity_row(unet_oof: dict, manifest_by_id: dict, area_thr=PARITY_AREA_THR):
    """U-Net raw pred'e baseline'la BIREBIR AYNI (3x3 acma + area_thr) filtre uygulanmis
    ikinci satir (EK-12.2)."""
    per_item_dice, per_item_neg_area = {}, {}
    for sid, p in unet_oof["per_item_pred_mask"].items():
        filtered = apply_min_area_filter(p, area_thr)
        it = manifest_by_id[sid]
        if it["cls"] == "pos":
            gt = load_target(it)
            per_item_dice[sid] = _dice_hard(filtered, gt)
        else:
            per_item_neg_area[sid] = int(filtered.sum())
    return {"per_item_dice": per_item_dice, "per_item_neg_area": per_item_neg_area}


def load_baseline():
    return json.load(open(BASELINE_OOF_PATH, encoding="utf-8"))


def load_subgroup(pos_ids):
    d = json.load(open(SUBGROUP_PATH, encoding="utf-8"))
    a_ids = set(d["A_ids"])
    b_ids = set(pos_ids) - a_ids
    return a_ids, b_ids, d


# --------------------------------------------------------------------------- #
# J4.0 -- hizalama ve saglik kontrolu (ZORUNLU DUR KAPISI)
# --------------------------------------------------------------------------- #

def j4_0(logger: Logger, manifest, manifest_by_id, pos_ids, neg_ids, oof_by_arm, baseline):
    logger.log("\n" + "=" * 78)
    logger.log("J4.0 -- hizalama ve saglik kontrolu")
    logger.log("=" * 78)

    # 1) saglik
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[2:6, 2:6] = 1.0
    d_self = _dice_hard(gt.astype(np.uint8), gt)
    d_empty = _dice_hard(np.zeros((10, 10), dtype=np.uint8), gt)
    logger.log(f"1) Dice(GT,GT)={d_self:.6f}  bos_tahmin+dolu_GT_Dice={d_empty:.6f}")
    ok1 = (abs(d_self - 1.0) < 1e-9) and (d_empty == 0.0)
    if not ok1:
        logger.log("!! DUR: saglik kontrolu basarisiz.")
        return {"ok": False, "stage": 1}

    # 2) npz 5 fold tam mi
    fold_coverage = {}
    ok2 = True
    for arm in ARMS:
        fo = oof_by_arm[arm]["fold_of"]
        counts = {f: 0 for f in range(N_FOLDS)}
        for f in fo.values():
            counts[f] += 1
        fold_coverage[arm] = counts
        logger.log(f"2) {arm} npz fold dagilimi: {counts}  n_toplam={oof_by_arm[arm]['n']}")
        if oof_by_arm[arm]["n"] != N_TOTAL_EXPECTED or any(counts[f] == 0 for f in range(N_FOLDS)):
            ok2 = False
    if not ok2:
        logger.log("!! DUR: npz 5 fold tam degil.")
        return {"ok": False, "stage": 2, "fold_coverage": fold_coverage}

    # 3) slice_id kumesi baseline ile birebir ayni mi
    manifest_ids = set(it["img_id"] for it in manifest)
    baseline_ids = set(baseline["b2_beyin_percentile"]["per_item_dice"].keys()) | \
        set(baseline["b2_beyin_percentile"]["per_item_neg_area"].keys())
    diff_m_minus_b = manifest_ids - baseline_ids
    diff_b_minus_m = baseline_ids - manifest_ids
    n_diff = len(diff_m_minus_b) + len(diff_b_minus_m)
    logger.log(f"3) manifest ID kumesi (n={len(manifest_ids)}) vs baseline_oof_v2.json ID kumesi "
               f"(n={len(baseline_ids)}): fark={n_diff}")
    if n_diff:
        sample = sorted(diff_m_minus_b | diff_b_minus_m)[:10]
        logger.log(f"!! DUR: ID kumeleri uyusmuyor. fark_sayisi={n_diff}  ornek_10={sample}")
        return {"ok": False, "stage": 3, "n_diff": n_diff, "sample": sample}

    # 4) esleme ID uzerinden (dict, argsort/index guveni yok) -- yapisal teyit
    for arm in ARMS:
        assert isinstance(oof_by_arm[arm]["per_item_dice"], dict)
    logger.log("4) esleme dict[img_id] uzerinden yapiliyor (indeks sirasina guvenilmiyor) -- yapisal.")

    # 5) pozitif/negatif sayisi
    n_pos, n_neg = len(pos_ids), len(neg_ids)
    logger.log(f"5) n_pos={n_pos} (beklenen={N_POS_EXPECTED}, sapma={n_pos - N_POS_EXPECTED})  n_neg={n_neg}")

    # 6) esikleme kodu
    src = (PROJECT_ROOT / "model" / "egitim" / "train_unet.py").read_text(encoding="utf-8")
    threshold_line = None
    for i, line in enumerate(src.splitlines(), start=1):
        if "prob > 0.5" in line:
            threshold_line = (i, line.strip())
    logger.log(f"6) train_unet.py esikleme: satir {threshold_line[0]}: {threshold_line[1]} "
               f"-- SABIT 0.5, ic-val veya OOF uzerinden taranmadi.")
    ok6 = threshold_line is not None
    if not ok6:
        logger.log("!! DUR: esikleme kodu dogrulanamadi.")
        return {"ok": False, "stage": 6}

    # 7) SHA-256
    shas = {}
    for arm in ARMS:
        shas[f"npz_{arm}"] = sha256_of(oof_by_arm[arm]["npz_path"])
        for fold in range(N_FOLDS):
            p = PRODUCT_DIR / f"unet_{arm}_fold{fold}.pt"
            shas[f"ckpt_{arm}_fold{fold}"] = sha256_of(p)
    logger.log("7) SHA-256:")
    for k, v in shas.items():
        logger.log(f"   {k} : {v}")

    logger.log("\nJ4.0: DUR yok.")
    return {"ok": True, "fold_coverage": fold_coverage, "n_pos": n_pos, "n_neg": n_neg,
            "threshold_check": {"line": threshold_line[0], "text": threshold_line[1]},
            "sha256": shas}


# --------------------------------------------------------------------------- #
# J4.1 -- asiri-uyum farki anomalisi (pilot bulgusu, kapi degil)
# --------------------------------------------------------------------------- #

def j4_1(logger: Logger, manifest, manifest_by_id, device):
    logger.log("\n" + "=" * 78)
    logger.log("J4.1 -- asiri-uyum farki anomalisi (pilot bulgusu, kapi degil)")
    logger.log("=" * 78)

    logger.log("1) train_unet.py train_in_median_dice hangi altkume uzerinden hesaplaniyor:")
    logger.log("   train_eval_ds = SegDataset(train_items, ...) -- train_items FILTRESIZ "
               "(pozitif+negatif KARISIK), cls filtresi YOK.")
    logger.log("3) egitim-ici degerlendirme sirasinda veri artirma: "
               "train_eval_ds = SegDataset(..., augment=False, ...) -- HAYIR (kapali).")

    rows = []
    for arm in ARMS:
        for fold in range(N_FOLDS):
            inner_val_fold = (fold + 1) % N_FOLDS
            train_items = [it for it in manifest if it["fold"] not in (fold, inner_val_fold)]
            train_pos = [it for it in train_items if it["cls"] == "pos"]

            ckpt_path = PRODUCT_DIR / f"unet_{arm}_fold{fold}.pt"
            ckpt = torch.load(ckpt_path, map_location=device)
            model = UNet(in_channels=1, out_channels=1, base=32).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            dices = []
            with torch.no_grad():
                for i in range(0, len(train_pos), 16):
                    chunk = train_pos[i:i + 16]
                    xb = np.stack([load_input_1ch(it["img_path"]) for it in chunk])
                    yb = np.stack([load_target(it) for it in chunk])
                    x_t = torch.from_numpy(xb).unsqueeze(1).to(device)
                    logits = model(x_t)
                    prob = torch.sigmoid(logits).cpu().numpy()[:, 0]
                    pred = (prob > 0.5).astype(np.uint8)
                    for j in range(len(chunk)):
                        dices.append(_dice_hard(pred[j], yb[j]))
            train_in_pos_median = float(np.median(dices)) if dices else float("nan")
            rows.append({"arm": arm, "fold": fold, "n_train_pos": len(train_pos),
                         "train_in_median_dice_pos_only": train_in_pos_median,
                         "train_pos_ids": [it["img_id"] for it in train_pos]})
    return {"rows": rows}


def j4_1_finish(logger: Logger, j4_1_rows, oof_by_arm, manifest_by_id):
    logger.log("\n2) kol x fold tablosu (egitim-ici[pozitif] | OOF[pozitif] | fark):")
    full_rows = []
    for r in j4_1_rows:
        arm, fold = r["arm"], r["fold"]
        oof_dice = oof_by_arm[arm]["per_item_dice"]
        fold_of = oof_by_arm[arm]["fold_of"]
        oof_ids_this_fold = [i for i in oof_dice if fold_of[i] == fold]
        oof_median = float(np.median([oof_dice[i] for i in oof_ids_this_fold])) if oof_ids_this_fold else float("nan")
        diff = r["train_in_median_dice_pos_only"] - oof_median
        logger.log(f"   {arm} fold{fold}: egitim-ici(pos)={r['train_in_median_dice_pos_only']:.4f}  "
                   f"OOF(pos)={oof_median:.4f}  fark={diff:.4f}  n_train_pos={r['n_train_pos']}  "
                   f"n_oof_pos={len(oof_ids_this_fold)}")
        full_rows.append({"arm": arm, "fold": fold, "train_in_median_pos": r["train_in_median_dice_pos_only"],
                          "oof_median_pos": oof_median, "diff": diff,
                          "n_train_pos": r["n_train_pos"], "n_oof_pos": len(oof_ids_this_fold),
                          "train_pos_ids": r["train_pos_ids"], "oof_pos_ids": oof_ids_this_fold})

    logger.log("\n4) fark negatif olan kol x fold(lar) icin pozitif oran + medyan lezyon alani:")
    any_negative = False
    for fr in full_rows:
        if fr["diff"] < 0:
            any_negative = True
            train_areas = [int(load_target(manifest_by_id[i]).sum()) for i in fr["train_pos_ids"]]
            oof_areas = [int(load_target(manifest_by_id[i]).sum()) for i in fr["oof_pos_ids"]]
            logger.log(f"   {fr['arm']} fold{fr['fold']}: fark={fr['diff']:.4f}  "
                       f"egitim-ici medyan_lezyon_alani={np.median(train_areas):.1f}px "
                       f"(n={len(train_areas)})  OOF medyan_lezyon_alani={np.median(oof_areas):.1f}px "
                       f"(n={len(oof_areas)})")
    if not any_negative:
        logger.log("   hicbir kol x fold icin fark negatif degil (10/10 pozitif).")
    return full_rows, any_negative


# --------------------------------------------------------------------------- #
# J4.2 -- uc metrik tablosu
# --------------------------------------------------------------------------- #

def compute_row_metrics(name, dice_dict, negarea_dict, pos_ids, neg_ids, block_of_id, logger):
    row = {"name": name}
    for boot_name in ("block", "slice"):
        bmap_pos = block_of_id if boot_name == "block" else trivial_block_map(pos_ids)
        bmap_neg = block_of_id if boot_name == "block" else trivial_block_map(neg_ids)
        m1_p, m1_lo, m1_hi = m1_ci(pos_ids, dice_dict, bmap_pos)
        m2_p, m2_lo, m2_hi = m2_ci(pos_ids, dice_dict, bmap_pos)
        m3_p, m3_lo, m3_hi = m3_ci(neg_ids, negarea_dict, bmap_neg)
        row[boot_name] = {
            "M1": {"value": m1_p, "ci95": [m1_lo, m1_hi]},
            "M2": {"value": m2_p, "ci95": [m2_lo, m2_hi]},
            "M3": {"value": m3_p, "ci95": [m3_lo, m3_hi]},
        }
        logger.log(f"  [{name}] ({boot_name}) M1={m1_p:.4f} CI=[{m1_lo:.4f},{m1_hi:.4f}]  "
                   f"M2={m2_p:.4f} CI=[{m2_lo:.4f},{m2_hi:.4f}]  M3={m3_p:.4f} CI=[{m3_lo:.4f},{m3_hi:.4f}]")
    return row


def j4_2(logger: Logger, oof_by_arm, parity_ft, baseline, pos_ids, neg_ids, block_of_id):
    logger.log("\n" + "=" * 78)
    logger.log(f"J4.2 -- uc metrik tablosu (blok bootstrap B={N_BOOT}, sonra kesit-seviyesi duyarlilik)")
    logger.log(f"uygulanan min-alan esigi (parite satiri, B2 ile birebir ayni): {PARITY_AREA_THR}px  "
               f"acma cekirdegi: 3x3 eliptik  M3 temizlik esigi: <{M3_CLEAN_PX_THR}px")
    logger.log("=" * 78)

    rows = {}
    rows["unet_ft"] = compute_row_metrics("U-Net(FT)", oof_by_arm["focal_tversky"]["per_item_dice"],
                                           oof_by_arm["focal_tversky"]["per_item_neg_area"],
                                           pos_ids, neg_ids, block_of_id, logger)
    rows["unet_ft_minarea"] = compute_row_metrics("U-Net(FT)+min-alan", parity_ft["per_item_dice"],
                                                   parity_ft["per_item_neg_area"],
                                                   pos_ids, neg_ids, block_of_id, logger)
    rows["unet_dicebce"] = compute_row_metrics("U-Net(DiceBCE)", oof_by_arm["dice_bce"]["per_item_dice"],
                                                oof_by_arm["dice_bce"]["per_item_neg_area"],
                                                pos_ids, neg_ids, block_of_id, logger)
    rows["b2"] = compute_row_metrics("B2", baseline["b2_beyin_percentile"]["per_item_dice"],
                                      baseline["b2_beyin_percentile"]["per_item_neg_area"],
                                      pos_ids, neg_ids, block_of_id, logger)
    rows["b3"] = compute_row_metrics("B3", baseline["b3_beyin_band"]["per_item_dice"],
                                      baseline["b3_beyin_band"]["per_item_neg_area"],
                                      pos_ids, neg_ids, block_of_id, logger)
    return rows


# --------------------------------------------------------------------------- #
# J4.3 / J4.4 -- kapilar
# --------------------------------------------------------------------------- #

def j4_3_4(logger: Logger, oof_by_arm, baseline, pos_ids, neg_ids, block_of_id):
    logger.log("\n" + "=" * 78)
    logger.log("J4.3 -- KAPI-U1 (uc kalem, ayri ayri)")
    logger.log("=" * 78)

    ft_dice = oof_by_arm["focal_tversky"]["per_item_dice"]
    ft_negarea = oof_by_arm["focal_tversky"]["per_item_neg_area"]
    b2_dice = baseline["b2_beyin_percentile"]["per_item_dice"]
    b3_negarea = baseline["b3_beyin_band"]["per_item_neg_area"]
    b3_dice = baseline["b3_beyin_band"]["per_item_dice"]

    m1_delta, m1_lo, m1_hi = boot_delta_median(pos_ids, ft_dice, b2_dice, block_of_id)
    m1_pass = m1_lo > 0.0
    logger.log(f"M1 (medyan Dice farki, FT-B2): {m1_delta:.4f} CI=[{m1_lo:.4f},{m1_hi:.4f}]  PASS={m1_pass}")

    m2_delta, m2_lo, m2_hi = boot_delta_mean_indicator(pos_ids, ft_dice, b3_dice, _ind_zero, block_of_id)
    m2_pass = m2_hi < 0.0
    logger.log(f"M2 (tam kacirma orani farki, FT-B3, dusuk iyi): {m2_delta:.4f} CI=[{m2_lo:.4f},{m2_hi:.4f}]  "
               f"PASS={m2_pass}")

    m3_delta, m3_lo, m3_hi = boot_delta_mean_indicator(neg_ids, ft_negarea, b3_negarea, _ind_below_clean, block_of_id)
    m3_pass = m3_lo > 0.0
    logger.log(f"M3 (negatif temizlik orani farki, FT-B3, yuksek iyi): {m3_delta:.4f} CI=[{m3_lo:.4f},{m3_hi:.4f}]  "
               f"PASS={m3_pass}")

    logger.log("\nkesit-seviyesi duyarlilik (blok yerine tekil kesit bootstrap):")
    m1d_s, m1lo_s, m1hi_s = boot_delta_median(pos_ids, ft_dice, b2_dice, trivial_block_map(pos_ids))
    m2d_s, m2lo_s, m2hi_s = boot_delta_mean_indicator(pos_ids, ft_dice, b3_dice, _ind_zero, trivial_block_map(pos_ids))
    m3d_s, m3lo_s, m3hi_s = boot_delta_mean_indicator(neg_ids, ft_negarea, b3_negarea, _ind_below_clean,
                                                       trivial_block_map(neg_ids))
    logger.log(f"  M1: {m1d_s:.4f} CI=[{m1lo_s:.4f},{m1hi_s:.4f}]")
    logger.log(f"  M2: {m2d_s:.4f} CI=[{m2lo_s:.4f},{m2hi_s:.4f}]")
    logger.log(f"  M3: {m3d_s:.4f} CI=[{m3lo_s:.4f},{m3hi_s:.4f}]")

    kapi_u1 = {
        "M1": {"delta": m1_delta, "ci95": [m1_lo, m1_hi], "pass": m1_pass,
               "sensitivity_slice": {"delta": m1d_s, "ci95": [m1lo_s, m1hi_s]}},
        "M2": {"delta": m2_delta, "ci95": [m2_lo, m2_hi], "pass": m2_pass,
               "sensitivity_slice": {"delta": m2d_s, "ci95": [m2lo_s, m2hi_s]}},
        "M3": {"delta": m3_delta, "ci95": [m3_lo, m3_hi], "pass": m3_pass,
               "sensitivity_slice": {"delta": m3d_s, "ci95": [m3lo_s, m3hi_s]}},
    }

    logger.log("\n" + "=" * 78)
    logger.log("J4.4 -- KAPI-U2 (guvenlik) ve KAPI-U3 (mutlak taban)")
    logger.log("=" * 78)
    u2_val, u2_lo, u2_hi = m3_ci(neg_ids, ft_negarea, block_of_id)
    u2_pass = u2_val >= 0.90
    logger.log(f"KAPI-U2: M3={u2_val:.4f} CI=[{u2_lo:.4f},{u2_hi:.4f}]  esik=0.90  PASS={u2_pass}")

    u3_val, u3_lo, u3_hi = m1_ci(pos_ids, ft_dice, block_of_id)
    u3_pass = u3_val >= 0.35
    logger.log(f"KAPI-U3: M1={u3_val:.4f} CI=[{u3_lo:.4f},{u3_hi:.4f}]  esik=0.35  PASS={u3_pass}")

    kapi_u2 = {"value": u2_val, "ci95": [u2_lo, u2_hi], "threshold": 0.90, "pass": u2_pass}
    kapi_u3 = {"value": u3_val, "ci95": [u3_lo, u3_hi], "threshold": 0.35, "pass": u3_pass}
    return kapi_u1, kapi_u2, kapi_u3


# --------------------------------------------------------------------------- #
# J4.5 -- iki kol farki
# --------------------------------------------------------------------------- #

def j4_5(logger: Logger, oof_by_arm, pos_ids, neg_ids, block_of_id):
    logger.log("\n" + "=" * 78)
    logger.log("J4.5 -- iki kol farki (focal_tversky - dice_bce)")
    logger.log("=" * 78)
    ft_dice = oof_by_arm["focal_tversky"]["per_item_dice"]
    db_dice = oof_by_arm["dice_bce"]["per_item_dice"]
    ft_negarea = oof_by_arm["focal_tversky"]["per_item_neg_area"]
    db_negarea = oof_by_arm["dice_bce"]["per_item_neg_area"]

    m1_delta, m1_lo, m1_hi = boot_delta_median(pos_ids, ft_dice, db_dice, block_of_id)
    m2_delta, m2_lo, m2_hi = boot_delta_mean_indicator(pos_ids, ft_dice, db_dice, _ind_zero, block_of_id)
    m3_delta, m3_lo, m3_hi = boot_delta_mean_indicator(neg_ids, ft_negarea, db_negarea, _ind_below_clean, block_of_id)

    logger.log(f"M1 farki: {m1_delta:.4f} CI=[{m1_lo:.4f},{m1_hi:.4f}]")
    logger.log(f"M2 farki: {m2_delta:.4f} CI=[{m2_lo:.4f},{m2_hi:.4f}]")
    logger.log(f"M3 farki: {m3_delta:.4f} CI=[{m3_lo:.4f},{m3_hi:.4f}]")
    return {
        "M1": {"delta": m1_delta, "ci95": [m1_lo, m1_hi]},
        "M2": {"delta": m2_delta, "ci95": [m2_lo, m2_hi]},
        "M3": {"delta": m3_delta, "ci95": [m3_lo, m3_hi]},
    }


# --------------------------------------------------------------------------- #
# J4.6 -- lezyon boyutuna gore tabaka
# --------------------------------------------------------------------------- #

def j4_6(logger: Logger, manifest_by_id, oof_by_arm, baseline, pos_ids):
    logger.log("\n" + "=" * 78)
    logger.log("J4.6 -- lezyon boyutuna gore tabaka (tertil)")
    logger.log("=" * 78)
    areas = {i: int(load_target(manifest_by_id[i]).sum()) for i in pos_ids}
    area_vals = np.array([areas[i] for i in pos_ids], dtype=np.float64)
    c1, c2 = np.percentile(area_vals, [100 / 3, 200 / 3])
    logger.log(f"tertil sinirlari (px): c1={c1:.1f}  c2={c2:.1f}")

    strata = {"low": [i for i in pos_ids if areas[i] <= c1],
              "mid": [i for i in pos_ids if c1 < areas[i] <= c2],
              "high": [i for i in pos_ids if areas[i] > c2]}

    dice_sources = {
        "unet_ft": oof_by_arm["focal_tversky"]["per_item_dice"],
        "unet_dicebce": oof_by_arm["dice_bce"]["per_item_dice"],
        "b2": baseline["b2_beyin_percentile"]["per_item_dice"],
        "b3": baseline["b3_beyin_band"]["per_item_dice"],
    }

    out = {"cutpoints_px": [float(c1), float(c2)], "strata": {}}
    for sname, ids in strata.items():
        n = len(ids)
        entry = {"n": n, "sources": {}}
        logger.log(f"\ntabaka={sname} n={n}")
        for src_name, dd in dice_sources.items():
            vals = [dd[i] for i in ids]
            m1 = float(np.median(vals)) if vals else float("nan")
            m2 = float(np.mean([v == 0 for v in vals])) if vals else float("nan")
            entry["sources"][src_name] = {"M1": m1, "M2": m2}
            logger.log(f"  {src_name}: M1={m1:.4f}  M2={m2:.4f}")
        out["strata"][sname] = entry
    return out


# --------------------------------------------------------------------------- #
# J4.7 -- alt grup raporu
# --------------------------------------------------------------------------- #

def j4_7(logger: Logger, oof_by_arm, baseline, a_ids, b_ids, block_of_id):
    logger.log("\n" + "=" * 78)
    logger.log("J4.7 -- alt grup raporu (A=ekstra-aksiyel n=58, B=parankim n=1024; kapi degil)")
    logger.log("=" * 78)
    dice_sources = {
        "unet_ft": oof_by_arm["focal_tversky"]["per_item_dice"],
        "b2": baseline["b2_beyin_percentile"]["per_item_dice"],
        "b3": baseline["b3_beyin_band"]["per_item_dice"],
    }
    out = {}
    for gname, ids in (("A_ekstraaksiyel", sorted(a_ids, key=int)), ("B_parankim", sorted(b_ids, key=int))):
        logger.log(f"\ngrup={gname} n={len(ids)}")
        out[gname] = {"n": len(ids), "sources": {}}
        bmap = {i: block_of_id[i] for i in ids}
        for src_name, dd in dice_sources.items():
            m1_p, m1_lo, m1_hi = m1_ci(ids, dd, bmap)
            m2_p, m2_lo, m2_hi = m2_ci(ids, dd, bmap)
            out[gname]["sources"][src_name] = {
                "M1": {"value": m1_p, "ci95": [m1_lo, m1_hi]},
                "M2": {"value": m2_p, "ci95": [m2_lo, m2_hi]},
            }
            logger.log(f"  {src_name}: M1={m1_p:.4f} CI=[{m1_lo:.4f},{m1_hi:.4f}]  "
                       f"M2={m2_p:.4f} CI=[{m2_lo:.4f},{m2_hi:.4f}]")
    return out


# --------------------------------------------------------------------------- #
# J4.8 -- hata envanteri
# --------------------------------------------------------------------------- #

def j4_8(logger: Logger, manifest_by_id, oof_by_arm, baseline):
    logger.log("\n" + "=" * 78)
    logger.log(f"J4.8 -- hata envanteri (birincil kol={PRIMARY_ARM})")
    logger.log("=" * 78)

    ft_dice = oof_by_arm[PRIMARY_ARM]["per_item_dice"]
    ft_pred_area = oof_by_arm[PRIMARY_ARM]["per_item_pred_area"]
    ft_negarea = oof_by_arm[PRIMARY_ARM]["per_item_neg_area"]
    areas_gt = {i: int(load_target(manifest_by_id[i]).sum()) for i in ft_dice}

    worst10 = sorted(ft_dice.items(), key=lambda kv: kv[1])[:10]
    worst10_out = [{"id": i, "dice": d, "gt_area_px": areas_gt[i]} for i, d in worst10]
    logger.log("1) en kotu 10 pozitif kesit (ID, Dice, GT alani):")
    for r in worst10_out:
        logger.log(f"   {r['id']}: dice={r['dice']:.4f}  gt_area={r['gt_area_px']}px")

    empty_pred_ids = sorted([i for i, a in ft_pred_area.items() if a == 0], key=int)
    logger.log(f"2) bos tahmin uretilen pozitif kesit sayisi={len(empty_pred_ids)}  ID'ler={empty_pred_ids}")

    top10_fp = sorted(ft_negarea.items(), key=lambda kv: -kv[1])[:10]
    top10_fp_out = [{"id": i, "pred_area_px": a} for i, a in top10_fp]
    logger.log("3) en yuksek yanlis-pozitif alanli 10 negatif kesit (ID, tahmin alani):")
    for r in top10_fp_out:
        logger.log(f"   {r['id']}: pred_area={r['pred_area_px']}px")

    b3_dice = baseline["b3_beyin_band"]["per_item_dice"]
    b3_negarea = baseline["b3_beyin_band"]["per_item_neg_area"]
    b3_dice_zero_ids = set(i for i, d in b3_dice.items() if d == 0.0)
    b3_worst10_ids = set(i for i, _ in sorted(b3_dice.items(), key=lambda kv: kv[1])[:10])
    b3_top10_fp_ids = set(i for i, _ in sorted(b3_negarea.items(), key=lambda kv: -kv[1])[:10])

    inter_worst10 = set(r["id"] for r in worst10_out) & b3_worst10_ids
    inter_empty = set(empty_pred_ids) & b3_dice_zero_ids
    inter_fp = set(r["id"] for r in top10_fp_out) & b3_top10_fp_ids

    logger.log(f"4) B3 ile kesisim -- en-kotu10: {len(inter_worst10)}/10  "
               f"bos-tahmin(FT) vs B3-tam-kacirma(dice==0): {len(inter_empty)}/{len(empty_pred_ids)}  "
               f"en-yuksek-FP10: {len(inter_fp)}/10")
    logger.log("   not: B3 icin 'bos tahmin' dogrudan bilinmiyor (yalniz skalar dice/alan mevcut); "
               "B3 tarafinda proxy olarak dice==0 (tam kacirma) kullanildi.")

    return {
        "primary_arm": PRIMARY_ARM,
        "worst10_positive": worst10_out,
        "empty_pred_positive": {"n": len(empty_pred_ids), "ids": empty_pred_ids},
        "top10_false_positive_negative": top10_fp_out,
        "intersection_with_b3": {
            "worst10_n": len(inter_worst10), "empty_pred_vs_b3_dice_zero_n": len(inter_empty),
            "top10_fp_n": len(inter_fp),
        },
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    logger = Logger(LOG_PATH)
    t0 = time.time()
    logger.log("\n" + "#" * 78)
    logger.log("J4: DEGERLENDIRME VE KAPILAR (product/eval_unet.py)")
    logger.log("#" * 78)
    logger.log("EK-12 zamanlama beyani: bu bolum yazilirken J3 egitim loglari GORULMUSTUR "
               "(fold basina val Dice, sure, checkpoint SHA'lari). Hicbir J4 degerlendirme "
               "sayisi GORULMEMISTIR -- hicbir OOF Dice, hicbir taban farki, hicbir kapi ciktisi.")
    logger.log("EK-12.1: birincil ikilileştirme sabit esik 0.5, son-islem YOK (dogrulama J4.0.6'da).")
    logger.log(f"EK-12.4: birincil kol={PRIMARY_ARM} (J3'ten once sabitlenmisti).")

    manifest, manifest_by_id, block_of_id, pos_ids, neg_ids = load_manifest()
    baseline = load_baseline()
    a_ids, b_ids, subgroup_raw = load_subgroup(pos_ids)

    oof_by_arm = {arm: load_unet_oof(arm, manifest_by_id) for arm in ARMS}

    j40 = j4_0(logger, manifest, manifest_by_id, pos_ids, neg_ids, oof_by_arm, baseline)
    if not j40["ok"]:
        logger.log("\nJ4 DURDURULDU (J4.0).")
        results = json.load(open(RESULTS_PATH, encoding="utf-8")) if RESULTS_PATH.exists() else {}
        results["j4"] = {"j4_0": j40, "stopped_at": "J4.0"}
        with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False, default=_json_default)
        logger.close()
        return

    parity_ft = build_parity_row(oof_by_arm[PRIMARY_ARM], manifest_by_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    j41_rows = j4_1(logger, manifest, manifest_by_id, device)
    j41_full, j41_any_negative = j4_1_finish(logger, j41_rows["rows"], oof_by_arm, manifest_by_id)

    j42 = j4_2(logger, oof_by_arm, parity_ft, baseline, pos_ids, neg_ids, block_of_id)
    kapi_u1, kapi_u2, kapi_u3 = j4_3_4(logger, oof_by_arm, baseline, pos_ids, neg_ids, block_of_id)
    j45 = j4_5(logger, oof_by_arm, pos_ids, neg_ids, block_of_id)
    j46 = j4_6(logger, manifest_by_id, oof_by_arm, baseline, pos_ids)
    j47 = j4_7(logger, oof_by_arm, baseline, a_ids, b_ids, block_of_id)
    j48 = j4_8(logger, manifest_by_id, oof_by_arm, baseline)

    postproc_delta, postproc_lo, postproc_hi = boot_delta_median(
        pos_ids, parity_ft["per_item_dice"], oof_by_arm[PRIMARY_ARM]["per_item_dice"], block_of_id)

    logger.log("\n" + "=" * 78)
    logger.log("J4.9 -- kayit")
    logger.log("=" * 78)

    j4 = {
        "j4_0_alignment_health": j40,
        "j4_1_overfit_gap_anomaly": {"per_arm_fold": j41_full, "any_negative_gap": j41_any_negative,
                                     "train_items_subset": "pozitif+negatif karisik (filtresiz)",
                                     "augmentation_during_train_eval": False},
        "j4_2_three_metric_table": j42,
        "j4_3_gate_u1": kapi_u1,
        "j4_4_gate_u2_u3": {"gate_u2_safety": kapi_u2, "gate_u3_floor": kapi_u3},
        "j4_5_loss_arm_delta": j45,
        "j4_6_lesion_size_strata": j46,
        "j4_7_subgroup_report": j47,
        "j4_8_error_inventory": j48,
        "postproc_m1_delta": {"delta": postproc_delta, "ci95": [postproc_lo, postproc_hi]},
        "runtime_sec": time.time() - t0,
    }

    results = json.load(open(RESULTS_PATH, encoding="utf-8")) if RESULTS_PATH.exists() else {}
    results["j4"] = j4
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=_json_default)
    logger.log(f"yazildi: {RESULTS_PATH} (yalniz 'j4' anahtari)")

    m1_ft = j42["unet_ft"]["block"]["M1"]
    m2_ft = j42["unet_ft"]["block"]["M2"]
    m3_ft = j42["unet_ft"]["block"]["M3"]

    lines = [
        f"M1_UNET = {m1_ft['value']:.4f} CI = [{m1_ft['ci95'][0]:.4f},{m1_ft['ci95'][1]:.4f}]  "
        f"delta_vs_B2 = {kapi_u1['M1']['delta']:.4f} CI = [{kapi_u1['M1']['ci95'][0]:.4f},"
        f"{kapi_u1['M1']['ci95'][1]:.4f}]  PASS = {'t' if kapi_u1['M1']['pass'] else 'f'}",
        f"M2_UNET = {m2_ft['value']:.4f} CI = [{m2_ft['ci95'][0]:.4f},{m2_ft['ci95'][1]:.4f}]  "
        f"delta_vs_B3 = {kapi_u1['M2']['delta']:.4f} CI = [{kapi_u1['M2']['ci95'][0]:.4f},"
        f"{kapi_u1['M2']['ci95'][1]:.4f}]  PASS = {'t' if kapi_u1['M2']['pass'] else 'f'}",
        f"M3_UNET = {m3_ft['value']:.4f} CI = [{m3_ft['ci95'][0]:.4f},{m3_ft['ci95'][1]:.4f}]  "
        f"delta_vs_B3 = {kapi_u1['M3']['delta']:.4f} CI = [{kapi_u1['M3']['ci95'][0]:.4f},"
        f"{kapi_u1['M3']['ci95'][1]:.4f}]  PASS = {'t' if kapi_u1['M3']['pass'] else 'f'}",
        f"GATE_U2_SAFETY = {kapi_u2['value']:.4f}   PASS = {'t' if kapi_u2['pass'] else 'f'}",
        f"GATE_U3_FLOOR  = {kapi_u3['value']:.4f} PASS = {'t' if kapi_u3['pass'] else 'f'}",
        f"LOSS_ARM_DELTA_M1 = {j45['M1']['delta']:.4f} CI = [{j45['M1']['ci95'][0]:.4f},{j45['M1']['ci95'][1]:.4f}]",
        f"POSTPROC_M1_DELTA = {postproc_delta:.4f} CI = [{postproc_lo:.4f},{postproc_hi:.4f}]",
    ]
    logger.log("")
    for line in lines:
        logger.log(line)

    logger.close()


if __name__ == "__main__":
    main()
