"""
model/cekirdek/confound_check.py -- girdinin "temiz BT kesiti mi, yoksa
uzerine isaretleme yapilmis (overlay) bir goruntu mu" oldugunu ayirt etme
mantigi.

Egitim verisindeki pozitif maskeler, overlay (uzerine renkli isaretleme
yapilmis) goruntulerden renk doygunluguna bakarak cikarildigi icin, modelin
gercekten BT piksellerinden mi ogrendigini yoksa isaretlemenin kendisinden mi
ogrendigini ayirmak gerekiyordu. Bu dosya o ayrimi dogrulayan testleri
(piksel seviyesinde ve model-cikti seviyesinde) icerir; load_colored_224 ve
region_stats fonksiyonlari ayrica model/ai-servisi/app.py tarafindan canli
girdi denetiminde kullanilir.

Hicbir model egitilmez; hicbir mevcut .npz/.pt/.json dosyasinin uzerine
yazilmaz.

Kosum (orijinal veri seti mevcutsa):
    python model/cekirdek/confound_check.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .eval_unet import (  # noqa: E402
    OUT_SIZE, N_BOOT, boot_delta_median, m1_ci, m2_ci, m3_ci, trivial_block_map,
    load_target, load_input_1ch, _dice_hard, _json_default,
)
from .preprocess import head_mask_from_gray, read_bgr  # noqa: E402
from .unet_model import UNet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_DIR = PROJECT_ROOT / "model" / "agirliklar"
MANIFEST_PATH = PROJECT_ROOT / "model" / "egitim" / "dataset_manifest.json"
RESULTS_PATH = PRODUCT_DIR / "unet_results.json"
LOG_PATH = PRODUCT_DIR / "unet_log.txt"
FT_OOF_NPZ = PRODUCT_DIR / "unet_oof_focal_tversky.npz"

PRIMARY_ARM = "focal_tversky"
N_FOLDS = 5
SAMPLE_N_POS = 50
SAMPLE_N_NEG = 50
SAMPLE_SEED = 42
SAT_DELTA_DUR_THR = 5.0  # /255 -- ham 0-255 S-kanal olceginde


class Logger:
    def __init__(self, path: Path):
        self.fh = open(path, "a", encoding="utf-8")

    def log(self, msg: str = ""):
        print(msg)
        self.fh.write(str(msg) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# --------------------------------------------------------------------------- #
# yukleme / bolge istatistikleri
# --------------------------------------------------------------------------- #

def load_manifest():
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    by_id = {it["img_id"]: it for it in manifest}
    block_of_id = {it["img_id"]: it["block_id"] for it in manifest}
    return manifest, by_id, block_of_id


def load_colored_224(img_path: str):
    """train_unet.py:load_input_1ch ile AYNI adim sirasi (oku->gri->kafa maskesi->
    maskeleme->resize), yalniz gri'ye COKMEDEN ONCE renk kanallarini korur --
    boylece 'modele giden goruntunun' renk icerigi (varsa) olculebilir."""
    bgr = read_bgr(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    head_mask, _ = head_mask_from_gray(gray)
    masked_bgr = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    masked_bgr[head_mask == 0] = 0
    resized_bgr = cv2.resize(masked_bgr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    head224 = cv2.resize(head_mask, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return resized_bgr, head224


def region_stats(bgr224: np.ndarray, mask: np.ndarray):
    if not mask.any():
        return None
    b = bgr224[:, :, 0].astype(np.int32)
    g = bgr224[:, :, 1].astype(np.int32)
    r = bgr224[:, :, 2].astype(np.int32)
    chdiff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    hsv = cv2.cvtColor(bgr224, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float64)
    gray = cv2.cvtColor(bgr224, cv2.COLOR_BGR2GRAY)
    return {
        "max_chdiff": int(chdiff[mask].max()),
        "sat_mean": float(sat[mask].mean()),
        "sat_p99": float(np.percentile(sat[mask], 99)),
        "gray_median": float(np.median(gray[mask])),
        "n_px": int(mask.sum()),
    }


def dir_template(paths: list) -> str:
    parents = set(str(Path(p).parent) for p in paths)
    if len(parents) == 1:
        return next(iter(parents)) + "\\{img_id}.png"
    return f"COKLU_DIZIN({len(parents)}): " + " | ".join(sorted(parents)[:3])


# --------------------------------------------------------------------------- #
# K1
# --------------------------------------------------------------------------- #

def k1(logger: Logger, manifest, manifest_by_id):
    logger.log("\n" + "=" * 78)
    logger.log("K1 -- girdi hatti denetimi")
    logger.log("=" * 78)

    logger.log("kod okuma: product/preprocess.py read_bgr(path_or_array) -- verilen yolu "
               "(np.fromfile+cv2.imdecode) DOGRUDAN okur, baska bir dizine yonlendirme yok.")
    logger.log("kod okuma: product/train_unet.py:100 load_input_1ch(img_path): "
               "bgr = read_bgr(img_path) -- img_path=it[\"img_path\"] (manifest'ten), "
               "satir 332: cache_in[it[\"img_id\"]] = load_input_1ch(it[\"img_path\"]).")

    pos_paths = [it["img_path"] for it in manifest if it["cls"] == "pos"]
    neg_paths = [it["img_path"] for it in manifest if it["cls"] == "neg"]
    dir_pos = dir_template(pos_paths)
    dir_neg = dir_template(neg_paths)
    logger.log(f"1) pozitif yukleme dizini sablonu : {dir_pos}")
    logger.log(f"1) negatif yukleme dizini sablonu  : {dir_neg}")
    dirs_differ = dir_pos != dir_neg
    logger.log(f"   dizinler farkli mi: {dirs_differ}")

    rng = random.Random(SAMPLE_SEED)
    pos_ids_all = sorted([it["img_id"] for it in manifest if it["cls"] == "pos"], key=int)
    neg_ids_all = sorted([it["img_id"] for it in manifest if it["cls"] == "neg"], key=int)
    sample_pos = rng.sample(pos_ids_all, SAMPLE_N_POS)
    sample_neg = rng.sample(neg_ids_all, SAMPLE_N_NEG)

    pos_records, neg_records = [], []
    for img_id in sample_pos:
        it = manifest_by_id[img_id]
        bgr224, head224 = load_colored_224(it["img_path"])
        gt224 = load_target(it) > 0
        mask_in = gt224
        mask_out = (head224 > 0) & (~gt224)
        s_in = region_stats(bgr224, mask_in)
        s_out = region_stats(bgr224, mask_out)
        pos_records.append({"id": img_id, "in": s_in, "out": s_out})

    for img_id in sample_neg:
        it = manifest_by_id[img_id]
        bgr224, head224 = load_colored_224(it["img_path"])
        s_all = region_stats(bgr224, head224 > 0)
        neg_records.append({"id": img_id, "all": s_all})

    sat_in_vals = [r["in"]["sat_mean"] for r in pos_records if r["in"]]
    sat_out_vals = [r["out"]["sat_mean"] for r in pos_records if r["out"]]
    sat_neg_vals = [r["all"]["sat_mean"] for r in neg_records if r["all"]]
    maxdiff_all = [r["in"]["max_chdiff"] for r in pos_records if r["in"]] + \
        [r["out"]["max_chdiff"] for r in pos_records if r["out"]] + \
        [r["all"]["max_chdiff"] for r in neg_records if r["all"]]

    sat_in_mean = float(np.mean(sat_in_vals))
    sat_out_mean = float(np.mean(sat_out_vals))
    sat_delta = sat_in_mean - sat_out_mean
    sat_neg_mean = float(np.mean(sat_neg_vals))
    channel_maxdiff = int(max(maxdiff_all))

    logger.log(f"\n2/3) n_pos_sample={len(pos_records)}  n_neg_sample={len(neg_records)}")
    logger.log(f"     pozitif maske-ici ortalama doygunluk (havuz-ortalamasi): {sat_in_mean:.4f}")
    logger.log(f"     pozitif maske-disi ortalama doygunluk (havuz-ortalamasi): {sat_out_mean:.4f}")
    logger.log(f"     fark (ici-disi): {sat_delta:.4f}")
    logger.log(f"     negatif ortalama doygunluk (havuz-ortalamasi): {sat_neg_mean:.4f}")
    logger.log(f"     kanal-arasi maksimum mutlak fark (tum orneklem, tum bolgeler): {channel_maxdiff}")

    gray_in_vals = np.array([r["in"]["gray_median"] for r in pos_records if r["in"]])
    gray_out_vals = np.array([r["out"]["gray_median"] for r in pos_records if r["out"]])
    gray_neg_vals = np.array([r["all"]["gray_median"] for r in neg_records if r["all"]])

    def med_iqr(a):
        if a.size == 0:
            return None
        return {"median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
                "q3": float(np.percentile(a, 75))}

    gray_summary = {"pos_in": med_iqr(gray_in_vals), "pos_out": med_iqr(gray_out_vals),
                    "neg_all": med_iqr(gray_neg_vals)}
    if dirs_differ:
        logger.log("\n5) gri yogunluk (medyan[Q1,Q3]) -- pozitif vs negatif havuz (orneklem):")
        logger.log(f"   pozitif (maske-ici) : {gray_summary['pos_in']}")
        logger.log(f"   pozitif (maske-disi): {gray_summary['pos_out']}")
        logger.log(f"   negatif (tum beyin) : {gray_summary['neg_all']}")

    halt = (sat_delta > SAT_DELTA_DUR_THR) or (channel_maxdiff != 0)
    logger.log(f"\n4) DUR kosulu: fark({sat_delta:.4f}) > {SAT_DELTA_DUR_THR} VEYA "
               f"kanal-fark({channel_maxdiff}) != 0  ->  DUR={halt}")
    if halt:
        logger.log("!! DUR: K1 basarisiz -- K2 kosulmaz.")

    return {
        "ok": not halt,
        "input_dir_pos": dir_pos, "input_dir_neg": dir_neg, "dirs_differ": dirs_differ,
        "sample_seed": SAMPLE_SEED, "n_pos_sample": len(pos_records), "n_neg_sample": len(neg_records),
        "sat_in_mean": sat_in_mean, "sat_out_mean": sat_out_mean, "sat_delta": sat_delta,
        "sat_neg_mean": sat_neg_mean, "channel_maxdiff": channel_maxdiff,
        "gray_intensity_summary": gray_summary,
        "halt": halt, "halt_threshold_sat_delta": SAT_DELTA_DUR_THR,
    }


# --------------------------------------------------------------------------- #
# K2
# --------------------------------------------------------------------------- #

def find_clean_counterpart(img_path: str):
    p = Path(img_path)
    parts = list(p.parts)
    parts_lower = [x.lower() for x in parts]
    if "overlay" in parts_lower:
        idx = parts_lower.index("overlay")
        candidate = Path(*parts[:idx], "PNG", *parts[idx + 1:])
    else:
        candidate = p
    return candidate if candidate.exists() else None


def k2(logger: Logger, manifest, manifest_by_id, block_of_id, device):
    logger.log("\n" + "=" * 78)
    logger.log("K2 -- eslestirilmis temiz/boyali cikarim")
    logger.log("=" * 78)

    ft_npz = np.load(FT_OOF_NPZ, allow_pickle=True)
    slice_id_arr = [str(s) for s in ft_npz["slice_id"]]
    fold_arr = ft_npz["fold"]
    fold_of_oof = dict(zip(slice_id_arr, [int(f) for f in fold_arr]))

    all_items = manifest
    matches = {}
    for it in all_items:
        cand = find_clean_counterpart(it["img_path"])
        if cand is not None:
            matches[it["img_id"]] = cand
    n_pos_total = sum(1 for it in manifest if it["cls"] == "pos")
    n_pos_matched = sum(1 for it in manifest if it["cls"] == "pos" and it["img_id"] in matches)
    n_neg_total = sum(1 for it in manifest if it["cls"] == "neg")
    n_neg_matched = sum(1 for it in manifest if it["cls"] == "neg" and it["img_id"] in matches)
    logger.log(f"1) boyasiz (OVERLAY olmayan) PNG karsiligi bulunan: pozitif {n_pos_matched}/{n_pos_total}  "
               f"negatif {n_neg_matched}/{n_neg_total}")

    if n_pos_matched == 0:
        logger.log("!! DUR: eslesen pozitif ID yok -- test yapilamaz.")
        return {"ok": False, "n_pos_matched": 0}

    # checkpoint'leri yukle (yalniz birincil kol)
    models = {}
    for fold in range(N_FOLDS):
        ckpt = torch.load(PRODUCT_DIR / f"unet_{PRIMARY_ARM}_fold{fold}.pt", map_location=device)
        m = UNet(in_channels=1, out_channels=1, base=32).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models[fold] = m

    def infer(img_path, fold):
        x = load_input_1ch(str(img_path))
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = models[fold](x_t)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        return (prob > 0.5).astype(np.uint8)

    t0 = time.time()
    dice_painted, dice_clean, gt_area = {}, {}, {}
    pos_matched_ids = [it["img_id"] for it in manifest if it["cls"] == "pos" and it["img_id"] in matches]
    for img_id in pos_matched_ids:
        it = manifest_by_id[img_id]
        fold = fold_of_oof[img_id]
        gt = load_target(it)
        pred_clean = infer(matches[img_id], fold)
        dice_clean[img_id] = _dice_hard(pred_clean, gt)
        gt_area[img_id] = int(gt.sum())

    ft_pred = ft_npz["pred"]
    ft_slice_id = slice_id_arr
    for i, sid in enumerate(ft_slice_id):
        if sid in dice_clean:
            it = manifest_by_id[sid]
            gt = load_target(it)
            dice_painted[sid] = _dice_hard(ft_pred[i], gt)
    logger.log(f"2) cikarim tamamlandi ({time.time() - t0:.1f}s)  n={len(pos_matched_ids)}")

    common_ids = sorted(set(dice_painted) & set(dice_clean), key=int)
    med_painted = float(np.median([dice_painted[i] for i in common_ids]))
    med_clean = float(np.median([dice_clean[i] for i in common_ids]))
    delta, lo, hi = boot_delta_median(common_ids, dice_painted, dice_clean, block_of_id, n_boot=N_BOOT)
    logger.log(f"3) Dice(boyali) medyan={med_painted:.4f}  Dice(boyasiz) medyan={med_clean:.4f}  "
               f"fark={delta:.4f} CI=[{lo:.4f},{hi:.4f}] (blok bootstrap B={N_BOOT})")

    n_total_miss = sum(1 for i in common_ids if dice_clean[i] == 0.0)
    frac_total_miss = n_total_miss / len(common_ids)
    logger.log(f"4) boyasiz girdide tam kacirma: n={n_total_miss}/{len(common_ids)}  oran={frac_total_miss:.4f}")

    # negatifler
    neg_matched_ids = [it["img_id"] for it in manifest if it["cls"] == "neg" and it["img_id"] in matches]
    neg_area_clean = {}
    for img_id in neg_matched_ids:
        fold = fold_of_oof.get(img_id)
        if fold is None:
            continue
        pred_clean = infer(matches[img_id], fold)
        neg_area_clean[img_id] = int(pred_clean.sum())
    neg_ids_with_area = sorted(neg_area_clean.keys(), key=int)
    m3_clean_val, m3_lo, m3_hi = m3_ci(neg_ids_with_area, neg_area_clean, block_of_id, n_boot=N_BOOT) \
        if neg_ids_with_area else (float("nan"), float("nan"), float("nan"))
    logger.log(f"5) boyasiz girdide negatif tahmin alani (n={len(neg_ids_with_area)}): "
               f"medyan={np.median(list(neg_area_clean.values())) if neg_area_clean else float('nan')}px  "
               f"M3(temizlik, <50px)={m3_clean_val:.4f} CI=[{m3_lo:.4f},{m3_hi:.4f}]")

    drops = sorted(common_ids, key=lambda i: dice_painted[i] - dice_clean[i], reverse=True)[:10]
    drop_rows = [{"id": i, "dice_painted": dice_painted[i], "dice_clean": dice_clean[i],
                 "gt_area_px": gt_area[i]} for i in drops]
    logger.log("6) en buyuk 10 dusus (ID, boyali Dice, boyasiz Dice, GT alani):")
    for r in drop_rows:
        logger.log(f"   {r['id']}: boyali={r['dice_painted']:.4f}  boyasiz={r['dice_clean']:.4f}  "
                   f"GT_alan={r['gt_area_px']}px")

    return {
        "ok": True,
        "n_pos_matched": n_pos_matched, "n_pos_total": n_pos_total,
        "n_neg_matched": n_neg_matched, "n_neg_total": n_neg_total,
        "dice_painted_median": med_painted, "dice_clean_median": med_clean,
        "delta": {"value": delta, "ci95": [lo, hi]},
        "clean_total_miss": {"n": n_total_miss, "frac": frac_total_miss},
        "clean_negative_pred_area": {"n": len(neg_ids_with_area),
                                     "m3_clean": {"value": m3_clean_val, "ci95": [m3_lo, m3_hi]}},
        "top10_drops": drop_rows,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    logger = Logger(LOG_PATH)
    t0 = time.time()
    logger.log("\n" + "#" * 78)
    logger.log("K1/K2: OVERLAY KONFAUND AYRIM TESTI (product/confound_check.py)")
    logger.log("#" * 78)
    logger.log("zamanlama beyani: bu bolum J4 sonuclari GORULDUKTEN SONRA yazilmistir; "
               "kapi degil, gecerlilik denetimidir.")

    manifest, manifest_by_id, block_of_id = load_manifest()

    k1_res = k1(logger, manifest, manifest_by_id)

    k2_res = None
    if k1_res["ok"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        k2_res = k2(logger, manifest, manifest_by_id, block_of_id, device)
    else:
        logger.log("\nK2 KOSULMADI (K1 DUR verdi).")

    results = json.load(open(RESULTS_PATH, encoding="utf-8")) if RESULTS_PATH.exists() else {}
    results["k1"] = k1_res
    if k2_res is not None:
        results["k2"] = k2_res
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=_json_default)
    keys_written = "k1, k2" if k2_res is not None else "k1"
    logger.log(f"\nyazildi: {RESULTS_PATH} (anahtar(lar): {keys_written})")

    logger.log("\n" + "=" * 78)
    logger.log("K3 -- kayit")
    logger.log("=" * 78)
    lines = [
        f"K1_SAT_IN  = {k1_res['sat_in_mean']:.4f}   K1_SAT_OUT = {k1_res['sat_out_mean']:.4f}   "
        f"K1_DELTA = {k1_res['sat_delta']:.4f}",
        f"K1_CHANNEL_MAXDIFF = {k1_res['channel_maxdiff']}",
        f"K1_INPUT_DIR_POS = {k1_res['input_dir_pos']}",
        f"K1_INPUT_DIR_NEG = {k1_res['input_dir_neg']}",
        f"K1_HALT = {'t' if k1_res['halt'] else 'f'}",
    ]
    if k2_res is not None:
        lines += [
            f"K2_N_MATCHED = {k2_res['n_pos_matched']}",
            f"K2_DICE_PAINTED = {k2_res['dice_painted_median']:.4f}  "
            f"K2_DICE_CLEAN = {k2_res['dice_clean_median']:.4f}",
            f"K2_DELTA = {k2_res['delta']['value']:.4f}  "
            f"CI = [{k2_res['delta']['ci95'][0]:.4f},{k2_res['delta']['ci95'][1]:.4f}]",
            f"K2_CLEAN_TOTAL_MISS = {k2_res['clean_total_miss']['frac']:.4f}",
        ]
    else:
        lines += ["K2_N_MATCHED = NA", "K2_DICE_PAINTED = NA  K2_DICE_CLEAN = NA",
                  "K2_DELTA = NA  CI = [NA,NA]", "K2_CLEAN_TOTAL_MISS = NA"]
    logger.log("")
    for line in lines:
        logger.log(line)

    logger.log(f"\ntoplam sure: {time.time() - t0:.1f}s")
    logger.close()


if __name__ == "__main__":
    main()
