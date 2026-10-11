"""
model/egitim/train_unet.py -- U-Net egitim betigi (Focal Tversky birincil
kol + Dice+BCE karsilastirma kolu), dataset_manifest.json'daki 5 fold ile.

GIRDI ON-ISLEME: model/cekirdek/preprocess.py'deki kafa-maskeli yontem
kullanilir (read_bgr -> gri -> kafa maskesi -> maske disini sifirla ->
resize). Egitim ve cikarim AYNI zinciri kullanir (bkz. load_input_1ch()),
tek fark egitimde 1 kanal [0,1] araligina normalize edilmesidir.

CV tasarimi: her disari-test fold'u f icin ic-val = fold (f+1)%5, ic-egitim =
kalan 3 fold (ic-ice/nested CV). Erken durdurma ve checkpoint secimi YALNIZ
ic-val Dice skoruna dayanir; disari-test fold'u f hicbir sekilde egitim/
checkpoint kararina karismaz (veri sizintisi yok). Boylece hem "en iyi val
Dice'a sahip checkpoint" hem de gercek out-of-fold test tahmini elde edilir.

Kosum (tek kol+fold, pilot):
    python model/egitim/train_unet.py --arm focal_tversky --fold 0 --max-epochs 2
Tam kosum (5 fold x 2 kol):
    python model/egitim/train_unet.py 2>&1 | Tee-Object -Append model/agirliklar/unet_log.txt

NOT: Bu betigin calismasi icin orijinal TEKNOFEST goruntu veri seti gerekir;
bu depoda yalniz kod ve dataset_manifest.json (fold/etiket atamalari) bulunur,
goruntu piksel verisi boyut/gizlilik nedeniyle depoya dahil edilmemistir.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.cekirdek.preprocess import head_mask_from_gray, read_bgr  # noqa: E402
from model.cekirdek.unet_losses import BCEDiceLoss, FocalTverskyLoss, soft_dice_score  # noqa: E402
from model.cekirdek.unet_model import UNet  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "model" / "egitim" / "dataset_manifest.json"
RESULTS_PATH = PROJECT_ROOT / "model" / "agirliklar" / "unet_results.json"
LOG_PATH = PROJECT_ROOT / "model" / "agirliklar" / "unet_log.txt"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"serialize edilemiyor: {type(o)}")


def median_iqr(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return "n/a"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.4f} [{q1:.4f},{q3:.4f}]"


OUT_SIZE = 224
N_FOLDS = 5
MAX_EPOCHS = 60
PATIENCE = 15
BATCH_SIZE = 16
LR = 1e-3
SEED = 42
OVERFIT_GAP_ALARM = 0.15
NUM_THREADS_PER_PROC = 12  # 2 kol paralel kosulacaksa (32 cekirdek), CPU rekabetini sinirlar
IC_ICE_CV_NOTU = (
    "U-Net ic-ice (nested) CV geregi 3/5 veriyle egitiliyor (ic-val icin 1 fold "
    "ayrildi). Bu nedenle, parametrelerini tum 4/5 egitim verisiyle secen basit "
    "bir taban modelle karsilastirildiginda U-Net daha az egitim verisiyle "
    "calisiyor -- bu asimetri duzeltilmez, sonucla birlikte oldugu gibi raporlanir."
)

ARM_SPECS = {
    "focal_tversky": lambda: FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75),
    "dice_bce": lambda: BCEDiceLoss(bce_weight=0.5, dice_weight=0.5),
}


def _verify_input_pipeline():
    """Guvenlik kontrolu: bu betik kafa-maskesi yontemi (head_mask_from_gray)
    disinda baska bir on-isleme (ornegin tam beyin maskesi cikaran bir
    "skull_strip" fonksiyonu) kullanmamali. Boyle bir fonksiyon bu dosyaya
    import edilmisse adi globals() icinde gorunur -- kontrol bunu arar."""
    if "skull_strip" in globals():
        print("!! DUR: skull_strip bu script'in ad alanina import edilmis -- girdi "
              "hattina beyin maskesi sizmis olabilir.")
        sys.exit(1)
    print("girdi hatti dogrulamasi: model/cekirdek/preprocess.py (read_bgr + head_mask_from_gray, "
          "KAFA maskeli) kullanildi. skull_strip bu script'e IMPORT EDILMEDI "
          "('skull_strip' in globals() == False).")


def load_input_1ch(img_path: str) -> np.ndarray:
    """prepare_image() ile AYNI adim sirasi (oku->gri->kafa maskesi->maskeleme->resize);
    yalniz kanal/normalize adimi segmentasyon icin 1-kanal [0,1] olarak uyarlanir."""
    bgr = read_bgr(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    head_mask, _ = head_mask_from_gray(gray)
    masked = gray.copy()
    masked[head_mask == 0] = 0
    resized = cv2.resize(masked, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0)


def load_target(item: dict) -> np.ndarray:
    if item["cls"] != "pos":
        return np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)
    gt = read_bgr(item["mask_path"])
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if gt.shape[:2] != (OUT_SIZE, OUT_SIZE):
        gt = cv2.resize(gt, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return (gt > 0).astype(np.float32)


def _augment(x2d: np.ndarray, y2d: np.ndarray, rng: np.random.Generator):
    """yatay cevirme + ±10 derece + ±%10 olcek. Yogunluk artirmasi yok."""
    if rng.random() < 0.5:
        x2d = np.ascontiguousarray(np.fliplr(x2d))
        y2d = np.ascontiguousarray(np.fliplr(y2d))
    angle = float(rng.uniform(-10, 10))
    scale = float(rng.uniform(0.9, 1.1))
    center = (OUT_SIZE / 2.0, OUT_SIZE / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    x2d = cv2.warpAffine(x2d, M, (OUT_SIZE, OUT_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    y2d = cv2.warpAffine(y2d, M, (OUT_SIZE, OUT_SIZE), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return x2d, y2d


class SegDataset(Dataset):
    def __init__(self, items: list, cache_in: dict, cache_tgt: dict, augment: bool, seed: int):
        self.items = items
        self.cache_in = cache_in
        self.cache_tgt = cache_tgt
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        x = self.cache_in[it["img_id"]]
        y = self.cache_tgt[it["img_id"]]
        if self.augment:
            x, y = _augment(x, y, self.rng)
        x_t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        y_t = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
        return x_t, y_t, it["img_id"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_n = 0
    dices = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = loss_fn(logits, y)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        with torch.no_grad():
            dices.extend(soft_dice_score(logits, y).cpu().numpy().tolist())
    return total_loss / max(total_n, 1), float(np.mean(dices)) if dices else float("nan")


def train_one(arm_name: str, fold: int, manifest: list, cache_in: dict, cache_tgt: dict,
              max_epochs: int, device: torch.device) -> dict:
    inner_val_fold = (fold + 1) % N_FOLDS
    train_items = [it for it in manifest if it["fold"] not in (fold, inner_val_fold)]
    val_items = [it for it in manifest if it["fold"] == inner_val_fold]
    test_items = [it for it in manifest if it["fold"] == fold]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = UNet(in_channels=1, out_channels=1, base=32).to(device)
    loss_fn = ARM_SPECS[arm_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_ds = SegDataset(train_items, cache_in, cache_tgt, augment=True, seed=SEED + fold)
    val_ds = SegDataset(val_items, cache_in, cache_tgt, augment=False, seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED + fold))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val_dice = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(max_epochs):
        te0 = time.time()
        train_loss, train_dice = _run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, val_dice = _run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - te0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_dice": val_dice, "sec": dt})
        print(f"    [{arm_name} fold={fold}] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} ({dt:.1f}s)")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"    [{arm_name} fold={fold}] erken durdurma: epoch {epoch} "
                  f"(en iyi epoch={best_epoch}, val_dice={best_val_dice:.4f})")
            break

    model.load_state_dict(best_state)

    ckpt_path = PROJECT_ROOT / "product" / f"unet_{arm_name}_fold{fold}.pt"
    torch.save({"state_dict": best_state, "arm": arm_name, "fold": fold,
                "best_epoch": best_epoch, "best_val_dice": best_val_dice}, ckpt_path)
    sha = _sha256(ckpt_path)
    print(f"    [{arm_name} fold={fold}] checkpoint: {ckpt_path.name}  SHA-256={sha}")

    # egitim-ici Dice (train_items, en iyi epoch agirliklariyla, artirma KAPALI)
    train_eval_ds = SegDataset(train_items, cache_in, cache_tgt, augment=False, seed=SEED)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    train_in_dices = []
    with torch.no_grad():
        for x, y, ids in train_eval_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            train_in_dices.extend(soft_dice_score(logits, y).cpu().numpy().tolist())
    train_in_median_dice = float(np.median(train_in_dices))

    # disari-test fold (asla egitim/checkpoint kararina karismadi) -- OOF tahmin
    test_ds = SegDataset(test_items, cache_in, cache_tgt, augment=False, seed=SEED)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    oof_records = []
    with torch.no_grad():
        for x, y, ids in test_loader:
            x_dev = x.to(device)
            logits = model(x_dev)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            for i, img_id in enumerate(ids):
                oof_records.append({
                    "img_id": img_id, "prob": prob[i, 0].astype(np.float16),
                    "pred": pred[i, 0], "is_positive": bool(y[i, 0].sum().item() > 0),
                })

    oof_dice_test_pos = [
        float(2 * (r["pred"].astype(bool) & (cache_tgt[r["img_id"]] > 0)).sum()
              / max(int(r["pred"].sum()) + int((cache_tgt[r["img_id"]] > 0).sum()), 1)
              if (int(r["pred"].sum()) + int((cache_tgt[r["img_id"]] > 0).sum())) > 0 else 1.0)
        for r in oof_records if r["is_positive"]
    ]
    oof_median_dice = float(np.median(oof_dice_test_pos)) if oof_dice_test_pos else float("nan")
    overfit_gap = train_in_median_dice - oof_median_dice

    return {
        "arm": arm_name, "fold": fold, "inner_val_fold": inner_val_fold,
        "n_train": len(train_items), "n_val": len(val_items), "n_test": len(test_items),
        "best_epoch": best_epoch, "best_val_dice": best_val_dice,
        "checkpoint": str(ckpt_path), "checkpoint_sha256": sha,
        "history": history, "runtime_sec": time.time() - t0,
        "train_in_median_dice": train_in_median_dice, "oof_median_dice_positive": oof_median_dice,
        "overfit_gap": overfit_gap, "overfit_alarm": bool(overfit_gap > OVERFIT_GAP_ALARM),
        "oof_records": oof_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_SPECS), default=None,
                    help="belirtilmezse her iki kol da kosulur")
    ap.add_argument("--fold", type=int, default=None, help="belirtilmezse 5 fold da kosulur")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()

    _verify_input_pipeline()
    print(IC_ICE_CV_NOTU)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(NUM_THREADS_PER_PROC)
        print(f"!! UYARI: GPU bulunamadi, CPU'da kosulacak "
              f"(threads={torch.get_num_threads()})")
    else:
        torch.backends.cudnn.benchmark = True
        print(f"cihaz: {device}  {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"cihaz: {device}  torch threads: {torch.get_num_threads()} "
          f"(paralel kosum icin surec basina sinirlandirildi: {NUM_THREADS_PER_PROC})")
    print("es-zamanlilik guvenligi: tohum (SEED=42) sabit; her fold icin model "
          "agirlik ilklendirmesi (torch.manual_seed(SEED)), DataLoader karistirma "
          "sirasi (generator=SEED+fold) ve artirma RNG'si (SEED+fold) yalniz "
          "kol/fold indeksine baglidir -- surece, zamanlamaya veya baska bir "
          "surecin es-zamanli calismasina baglilik yoktur. Paralel kosum sonucu "
          "etkilemez (yalniz sure kisalir).")

    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    t0 = time.time()
    cache_in, cache_tgt = {}, {}
    for it in manifest:
        cache_in[it["img_id"]] = load_input_1ch(it["img_path"])
        cache_tgt[it["img_id"]] = load_target(it)
    print(f"onbellek yuklendi: {len(cache_in)} goruntu ({time.time() - t0:.1f}s)")

    arms = [args.arm] if args.arm else list(ARM_SPECS)
    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))

    all_results = {}
    for arm_name in arms:
        fold_results = []
        for f in folds:
            print(f"\n{'=' * 78}\nJ3: kol={arm_name} fold={f} (ic-val fold={(f + 1) % N_FOLDS})\n{'=' * 78}")
            res = train_one(arm_name, f, manifest, cache_in, cache_tgt, args.max_epochs, device)
            fold_results.append(res)
            print(f"  fold {f} ozet: en iyi epoch={res['best_epoch']} val_dice={res['best_val_dice']:.4f} "
                  f"egitim-ici medyan Dice={res['train_in_median_dice']:.4f} "
                  f"OOF medyan Dice(pozitif)={res['oof_median_dice_positive']:.4f} "
                  f"asiri-uyum farki={res['overfit_gap']:.4f}"
                  + ("  !! ASIRI-UYUM UYARISI (>0.15)" if res["overfit_alarm"] else ""))
        all_results[arm_name] = fold_results

    # --- OOF npz yaz (mevcut dosyanin uzerine yazma -- varsa .bak'a tasi) ---
    for arm_name, fold_results in all_results.items():
        if len(fold_results) < N_FOLDS:
            print(f"\n[{arm_name}] uyari: {len(fold_results)}/{N_FOLDS} fold kosuldu -- "
                  f"npz TAM DEGIL, yazilmiyor (kismi pilot kosumu).")
            continue
        npz_path = PROJECT_ROOT / "product" / f"unet_oof_{arm_name}.npz"
        if npz_path.exists():
            bak = npz_path.with_suffix(f".bak{int(time.time())}.npz")
            shutil.move(str(npz_path), str(bak))
            print(f"mevcut npz korundu: {bak}")
        all_records = [r for fr in fold_results for r in fr["oof_records"]]
        np.savez_compressed(
            npz_path,
            prob=np.stack([r["prob"] for r in all_records]),
            pred=np.stack([r["pred"] for r in all_records]),
            slice_id=np.array([r["img_id"] for r in all_records]),
            is_positive=np.array([r["is_positive"] for r in all_records]),
        )
        print(f"yazildi: {npz_path}  (n={len(all_records)})")

    # --- unet_results.json'a j3 anahtari (mevcut anahtarlara dokunulmaz) ---
    results = {}
    if RESULTS_PATH.exists():
        results = json.load(open(RESULTS_PATH, encoding="utf-8"))

    def _strip(fr):
        return {k: v for k, v in fr.items() if k != "oof_records"}

    j3 = results.get("j3", {})
    for arm_name, fold_results in all_results.items():
        j3[arm_name] = {"fold_results": [_strip(fr) for fr in fold_results]}
    j3["device"] = str(device)
    j3["num_threads_per_proc"] = NUM_THREADS_PER_PROC
    j3["ic_ice_cv_notu"] = IC_ICE_CV_NOTU
    j3["total_runtime_sec"] = j3.get("total_runtime_sec", 0.0) + (time.time() - t0)
    results["j3"] = j3
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\nyazildi: {RESULTS_PATH}")
    print(f"\nJ3 tamamlandi (kol(lar)={arms}, fold(lar)={folds}). toplam sure={time.time() - t0:.1f}s cihaz={device}")


if __name__ == "__main__":
    main()
