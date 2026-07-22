#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
產生第四章「質化視覺對比圖」三張：thin / bold / relaxed。

版面（對齊論文圖說）：
    上半列 = CircleFont v2（手寫）
    下半列 = NotoSansJP-ExtraLight（電腦）
    每個字兩欄：原始字形 | 優化結果

輸出：figures/thin_comparison.{png,pdf}
      figures/bold_comparison.{png,pdf}
      figures/relax_comparison.{png,pdf}
產完後把 png/pdf 複製到  資工所論文/fig/  即可（論文用 png 或 pdf 皆可）。

執行位置：WSL 的  ~/FontCLIP_mine/  目錄底下
    python3 make_comparison_grids.py

相依套件：pip install pillow matplotlib numpy
================================================================
本版已「鎖定」你實際的輸出結構，正常情況不需要改任何路徑：

  output_v3_direction/{字型}/{prompt}_{字}/{字型}/{設定長名}/
      ├── video-png/iter0000.png   ← 原始（第 0 迭代）
      ├── video-png/iter0200.png   ← 優化結果（最後一張 iter）
      └── output-svg/output.svg    ← 優化後向量結果（本版不用）

原始與優化「兩邊都取 diffvg 自己 render 的 PNG」，
所以不需要另外指定 .ttf 字型檔，兩張圖的線條風格也一致。
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

# 優化結果的根目錄（run_test4808_v3.py 的輸出資料夾）
OPT_ROOT = "output_v3_direction"

# 資料夾使用的字型名（就是 OPT_ROOT 底下第一層的資料夾名）
FONT_DIRS = ["CircleFont_v2", "NotoSansJP-ExtraLight"]
# 圖左側顯示用的短字型名（全名寫在 LaTeX 圖說即可，這裡短一點免得擠）
FONT_TAGS = ["個人手寫字體", "NotoSansJP"]

# 字集：主文三張共用同一組 10 字（跨提示詞可比）；其餘 10 字產附錄圖。
# 兩組都橫跨三種複雜度（開放少筆畫 / 開放多筆畫 / 封閉輪廓），合起來 = 完整 20 字。
MAIN_CHARS     = list("一二三十人大木日月口")   # 主文用（含正文點名的字：一二三人大、月）
APPENDIX_CHARS = list("小山川土水火力又子田")   # 附錄用（其餘 10 字）

# 三個提示詞：label = 顯示用（relaxed）；folder = 資料夾實際用的（relax）
PROMPTS = [
    dict(label="纖細",    folder="thin"),
    dict(label="厚重",    folder="bold"),
    dict(label="隨性", folder="relax"),
]

# 自動組出 6 張圖：3 主文 + 3 附錄
#   主文輸出：thin_comparison / bold_comparison / relax_comparison
#   附錄輸出：thin_comparison_appendix / bold_comparison_appendix / relax_comparison_appendix
FIGS = {}
for _p in PROMPTS:
    FIGS[_p["label"]] = dict(
        label=_p["label"], folder=_p["folder"],
        chars=MAIN_CHARS, out=f"{_p['folder']}_comparison")
    FIGS[_p["label"] + "_appendix"] = dict(
        label=_p["label"], folder=_p["folder"],
        chars=APPENDIX_CHARS, out=f"{_p['folder']}_comparison_appendix")

OUT_DIR   = "figures"          # 產出資料夾
RENDER_PX = 256                # 每格統一縮放到的解析度
CELL_IN   = 0.9                # 每格英吋大小（控制整張圖尺寸）
# matplotlib 標籤用中文字型（WSL 可直接讀 Windows 字型）
CJK_LABEL_TTF = "/mnt/c/Windows/Fonts/msjh.ttc"
# ===================================================================


# ---- 讓 matplotlib 中文標籤正常顯示 ----
from matplotlib import font_manager as fm
if os.path.exists(CJK_LABEL_TTF):
    fm.fontManager.addfont(CJK_LABEL_TTF)
    plt.rcParams["font.family"] = fm.FontProperties(fname=CJK_LABEL_TTF).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _case_dir(font_dir, folder_prompt, ch):
    """
    回傳某 (字型, 提示詞, 字) 的「設定長名」資料夾路徑，找不到回傳 None。
    結構：OPT_ROOT/{font_dir}/{prompt}_{ch}/{font_dir}/{設定長名}/
    """
    parent = os.path.join(OPT_ROOT, font_dir, f"{folder_prompt}_{ch}", font_dir)
    if not os.path.isdir(parent):
        return None
    subs = [d for d in glob.glob(os.path.join(parent, "*")) if os.path.isdir(d)]
    return subs[0] if subs else None


def _iter_png(case_dir, which):
    """
    which="first" → 第 0 迭代（原始）；which="last" → 最後一張 iter（優化結果）。
    """
    pngs = glob.glob(os.path.join(case_dir, "video-png", "iter*.png"))
    if not pngs:
        return None

    def it(p):
        m = re.search(r"iter(\d+)\.png$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    pngs.sort(key=it)
    return pngs[0] if which == "first" else pngs[-1]


def load_gray(path, px=RENDER_PX):
    """讀圖 → 灰階 → 正方置中 → 縮放到 px。"""
    img = Image.open(path).convert("L")
    img = ImageOps.pad(img, (px, px), color=255, centering=(0.5, 0.5))
    return np.asarray(img)


def blank(px=RENDER_PX):
    return np.full((px, px), 240, dtype=np.uint8)


def build_figure(spec):
    label   = spec["label"]
    folder  = spec["folder"]
    chars   = spec["chars"]
    n       = len(chars)

    # 版面：每個字型拆成「原始 / 優化」兩列 → 共 4 列；每個字一欄 → n 欄。
    # 欄數只有 n（不是 2n），字會大一倍，也不會擠成長條。
    states  = ["原始", label]                 # 一個字型內的兩列
    nrows   = len(FONT_DIRS) * len(states)    # = 4
    ncols   = n

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(CELL_IN * ncols + 1.4, CELL_IN * nrows + 0.3),
        squeeze=False,
    )

    missing = []
    for fi, (font_dir, tag) in enumerate(zip(FONT_DIRS, FONT_TAGS)):
        for j, ch in enumerate(chars):
            cdir  = _case_dir(font_dir, folder, ch)
            imgs  = {
                "原始": _iter_png(cdir, "first") if cdir else None,
                label:  _iter_png(cdir, "last")  if cdir else None,
            }
            for si, st in enumerate(states):
                r  = fi * len(states) + si
                ax = axes[r][j]
                p  = imgs[st]
                ax.imshow(load_gray(p) if p else blank(),
                          cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
                if p is None:
                    missing.append((font_dir, ch, st))
                # 頂端一次標字
                if r == 0:
                    ax.set_title(ch, fontsize=14, pad=4)
                # 左側水平標「字型 / 原始或優化」，不旋轉、不擠
                if j == 0:
                    ax.set_ylabel(f"{tag}\n{st}", fontsize=9, rotation=0,
                                  ha="right", va="center", labelpad=10)

    # 不加總標題，避免和 LaTeX 圖說重複
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.08)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, spec["out"] + ".png")
    pdf = os.path.join(OUT_DIR, spec["out"] + ".pdf")
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
        print(f"⚠ 找不到優化輸出根目錄 OPT_ROOT={OPT_ROOT}（請在 ~/FontCLIP_mine/ 底下執行）")
        return
    for spec in FIGS.values():
        build_figure(spec)
    print("\n完成。請把 figures/ 下的 *_comparison.png（或 .pdf）複製到  資工所論文/fig/，"
          "再到第四章取消對應 \\includegraphics 的註解。")


if __name__ == "__main__":
    main()
