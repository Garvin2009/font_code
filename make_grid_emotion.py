#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
產生第四章「情緒屬性探測實驗」質化視覺對比圖一張：emotion_comparison。

版面（對齊 make_comparison_grids.py 的風格，緊湊、灰階、無灰底格線）：
    上半 4 列 = 個人手寫字體(CircleFont v2)：原始 / 憤怒 / 喜悅 / 悲傷
    下半 4 列 = NotoSansJP-ExtraLight       ：原始 / 憤怒 / 喜悅 / 悲傷
    每一欄 = 一個字（人 山 口 日 月 學 離 讀）

情緒提示詞：
    憤怒 angry → "angry, aggressive handwriting"
    喜悅 happy → "happy, cheerful handwriting"
    悲傷 sad   → "sad, sorrowful handwriting"
參考基準：formal, legible font

輸出：figures/emotion_comparison.{png,pdf}
      產完後把 png（或 pdf）複製到  資工所論文/fig/  → 第四章圖 fig:emotion_grid 用
      \includegraphics{fig/emotion_comparison.png}

執行位置：WSL 的  ~/FontCLIP_mine/  目錄底下
    python3 make_grid_emotion.py

相依套件：pip install pillow matplotlib numpy
================================================================
鎖定情緒實驗的輸出結構，正常情況不需要改任何路徑：

  output_emotion/{字型}/{emotion}_{字}/{字型}/{設定長名}/
      ├── video-png/iter0000.png   ← 原始（第 0 迭代）
      └── video-png/iter0200.png   ← 優化結果（最後一張 iter）

原始與優化「兩邊都取 diffvg 自己 render 的 PNG」，
兩張圖線條風格一致，不需另外指定 .ttf。
================================================================
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# ============================ 使用者設定 ============================

# 情緒實驗輸出根目錄（run_test_emotion.py 的輸出資料夾）
OPT_ROOT = "output_emotion"

# 資料夾使用的字型名（OPT_ROOT 底下第一層的資料夾名）
FONT_DIRS = ["CircleFont_v2", "NotoSansJP-ExtraLight"]
# 圖左側顯示用的短字型名（全名寫在 LaTeX 圖說即可）
FONT_TAGS = ["個人手寫字體", "NotoSansJP"]

# 8 個實驗字（跨情緒可比）
CHARS = list("人山口日月學離讀")

# 情緒：label = 圖上顯示；folder = 資料夾實際前綴
EMOTIONS = [
    dict(label="憤怒", folder="angry"),
    dict(label="喜悅", folder="happy"),
    dict(label="悲傷", folder="sad"),
]

OUT_NAME  = "emotion_comparison"   # 產出檔名
OUT_DIR   = "figures"              # 產出資料夾
RENDER_PX = 256                    # 每格統一縮放到的解析度
CELL_IN   = 0.62                   # 每格英吋大小（8 列 × 8 欄，調小以免整張過大）
CJK_LABEL_TTF = "/mnt/c/Windows/Fonts/msjh.ttc"
# ===================================================================


# ---- 讓 matplotlib 中文標籤正常顯示 ----
from matplotlib import font_manager as fm
if os.path.exists(CJK_LABEL_TTF):
    fm.fontManager.addfont(CJK_LABEL_TTF)
    plt.rcParams["font.family"] = fm.FontProperties(fname=CJK_LABEL_TTF).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _case_dir(font_dir, emotion_folder, ch):
    """
    回傳某 (字型, 情緒, 字) 的「設定長名」資料夾路徑，找不到回傳 None。
    結構：OPT_ROOT/{font_dir}/{emotion}_{ch}/{font_dir}/{設定長名}/
    """
    parent = os.path.join(OPT_ROOT, font_dir, f"{emotion_folder}_{ch}", font_dir)
    if not os.path.isdir(parent):
        return None
    subs = [d for d in glob.glob(os.path.join(parent, "*")) if os.path.isdir(d)]
    return subs[0] if subs else None


def _iter_png(case_dir, which):
    """which="first" → 第 0 迭代（原始）；which="last" → 最後一張 iter（優化結果）。"""
    if case_dir is None:
        return None
    pngs = glob.glob(os.path.join(case_dir, "video-png", "iter*.png"))
    if not pngs:
        return None

    def it(p):
        m = re.search(r"iter(\d+)\.png$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    pngs.sort(key=it)
    return pngs[0] if which == "first" else pngs[-1]


def load_gray(path, px=RENDER_PX):
    img = Image.open(path).convert("L")
    img = ImageOps.pad(img, (px, px), color=255, centering=(0.5, 0.5))
    return np.asarray(img)


def blank(px=RENDER_PX):
    return np.full((px, px), 240, dtype=np.uint8)


def build_figure():
    n = len(CHARS)
    # 每個字型 4 列：原始 + 3 情緒；兩字型 → 8 列。每字一欄 → n 欄。
    states = ["原始"] + [e["label"] for e in EMOTIONS]     # 4
    nrows  = len(FONT_DIRS) * len(states)                  # 8
    ncols  = n                                             # 8

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(CELL_IN * ncols + 1.6, CELL_IN * nrows + 0.3),
        squeeze=False,
    )

    missing = []
    for fi, (font_dir, tag) in enumerate(zip(FONT_DIRS, FONT_TAGS)):
        for j, ch in enumerate(CHARS):
            # 原始：任取一情緒（angry）的第 0 迭代即可
            orig = _iter_png(_case_dir(font_dir, EMOTIONS[0]["folder"], ch), "first")
            imgs = {"原始": orig}
            for e in EMOTIONS:
                imgs[e["label"]] = _iter_png(_case_dir(font_dir, e["folder"], ch), "last")

            for si, st in enumerate(states):
                r  = fi * len(states) + si
                ax = axes[r][j]
                p  = imgs.get(st)
                ax.imshow(load_gray(p) if p else blank(),
                          cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
                if p is None:
                    missing.append((font_dir, ch, st))
                if r == 0:
                    ax.set_title(ch, fontsize=13, pad=4)
                if j == 0:
                    ax.set_ylabel(f"{tag}\n{st}", fontsize=8, rotation=0,
                                  ha="right", va="center", labelpad=10)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.08)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, OUT_NAME + ".png")
    pdf = os.path.join(OUT_DIR, OUT_NAME + ".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {png}  /  {pdf}")
    if missing:
        print(f"  ⚠ 有 {len(missing)} 格找不到 iter png（已畫灰色佔位）：")
        for m in missing:
            print("     -", m)


def main():
    if not os.path.isdir(OPT_ROOT):
        print(f"⚠ 找不到情緒輸出根目錄 OPT_ROOT={OPT_ROOT}（請在 ~/FontCLIP_mine/ 底下執行）")
        return
    build_figure()
    print("\n完成。請把 figures/emotion_comparison.png（或 .pdf）複製到  資工所論文/fig/。")


if __name__ == "__main__":
    main()
