"""
model/cekirdek/preprocess.py -- egitim ve cikarimda ortak on-isleme modulu.

NOT: Goruntu dogrudan (letterbox'siz) 224x224'e resize edilir. Letterbox
(oran koruyarak + siyah kenar boşluğu ekleyerek boyutlandırma) kasıtlı olarak
kullanılmadı: egitim maskeleri de dogrudan resize ile uretildigi icin, girdiye
letterbox uygulayip maskeye uygulamamak goruntu/maske hizasini bozardi.

Egitim ve uretim AYNI fonksiyonu (prepare_image) cagirir. Sira SABITTIR:
  1) oku (np.fromfile + cv2.imdecode, IMREAD_COLOR) -- Turkce yol destegi
  2) gri seviyeye cevir
  3) kafa maskesi: Otsu esigi + en buyuk bagli bilesen + delik doldurma
     (_build_head_mask_bgr)
  4) kafa maskesi disini sifirla
  5) out_size'a dogrudan bilinear resize (letterbox yok)
  6) /255 + ImageNet normalize (3 kanala kopyalanir, siniflandirici RGB bekler)

prepare_mask(): dogrudan out_size'a INTER_NEAREST resize, normalize yok.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _build_head_mask_bgr(bgr: np.ndarray):
    """Gri-tonlu goruntude Otsu esigi ustu bolgenin en buyuk bagli bileseni + delik
    doldurma. Donus: (ikili maske uint8 0/255, otsu esigi)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if n_comp <= 1:
        return np.zeros(gray.shape, dtype=np.uint8), float(thr)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    head = (labels == largest).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(head)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled, float(thr)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_bgr(path_or_array):
    """Turkce yollar icin np.fromfile + cv2.imdecode. ndarray verilirse degistirmeden doner."""
    if isinstance(path_or_array, np.ndarray):
        return path_or_array
    data = np.fromfile(str(path_or_array), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path_or_array)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"decode edilemedi: {path_or_array}")
    return bgr


def head_mask_from_gray(gray: np.ndarray):
    """Bolum E'deki AYNI yontem (Otsu + en buyuk bagli bilesen + delik doldurma).
    build_head_mask BGR bekler; gri kanal 3'e kopyalanarak (R=G=B) BGR2GRAY donusumu
    ayni degerleri geri verir, boylece sonuc bit-bit aynidir."""
    bgr3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask, thr = _build_head_mask_bgr(bgr3)
    return mask, thr


def prepare_image(path_or_array, out_size: int = 224, return_head_mask: bool = False,
                   return_otsu_thr: bool = False):
    """Egitim ve uretimde ORTAK on-isleme (sira sabit, bkz. modul docstring).

    Donus: torch.FloatTensor [3,out_size,out_size] (ImageNet normalize).
    return_head_mask=True  -> ek olarak head_mask (out_size,out_size) uint8 (0/255) doner.
    return_otsu_thr=True   -> ek olarak otsu esigi (float) doner.
    """
    bgr = read_bgr(path_or_array)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    head_mask, head_thr = head_mask_from_gray(gray)
    masked = gray.copy()
    masked[head_mask == 0] = 0

    resized = cv2.resize(masked, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.astype(np.float32)).permute(2, 0, 1)

    result = [tensor]
    if return_head_mask:
        head_mask_out = cv2.resize(head_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        result.append(head_mask_out)
    if return_otsu_thr:
        result.append(head_thr)
    return tuple(result) if len(result) > 1 else result[0]


def prepare_mask(mask_or_path, out_size: int = 224) -> np.ndarray:
    """Dogrudan out_size'a INTER_NEAREST resize, normalize yok, letterbox yok."""
    mask = read_bgr(mask_or_path) if isinstance(mask_or_path, (str, Path)) else mask_or_path
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    out = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return out
