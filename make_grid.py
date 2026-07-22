"""
make_grid.py
把 svg_collect/ 裡的 SVG 渲染成一張大對照圖。

排列方式（6 列 × 20 行）：
  列 = 字型 × prompt（NotoSansJP: thin/bold/relax，CircleFont: thin/bold/relax）
  行 = 20 個字符（一二三十人大小山川土木水火日月口田力又子）

輸出：svg_grid.png（與本腳本同目錄）

需求：
  pip install cairosvg pillow
  （cairosvg 在 Ubuntu 上可能需要：sudo apt install libcairo2）
"""

import sys
from pathlib import Path

# ── 設定 ──────────────────────────────────────────────
BASE      = Path(__file__).parent
SVG_DIR   = BASE / "svg_collect/output_v3_direction"
OUT_FILE  = BASE / "svg_collect/svg_grid.png"

CHARS   = list("一二三十人大小山川土木水火日月口田力又子")
PROMPTS = ["thin", "bold", "relax"]
FONTS   = [
    ("NotoSansJP-ExtraLight", "Noto"),
    ("CircleFont_v2",         "Circle"),
]

CELL_W, CELL_H = 160, 160   # 每格大小（px）
LABEL_W        = 140         # 左側列標籤寬度
HEADER_H       = 50          # 上方字符標籤高度
PADDING        = 4           # 格間距

ROWS = [(fn, p) for fn, _ in FONTS for p in PROMPTS]   # 6 列
COLS = CHARS                                             # 20 行

IMG_W = LABEL_W  + len(COLS) * (CELL_W + PADDING)
IMG_H = HEADER_H + len(ROWS) * (CELL_H + PADDING)

# ── 轉換 SVG → PIL Image ──────────────────────────────
def svg_to_pil(svg_path: Path, size: int):
    """用 cairosvg 把 SVG 轉成 PIL Image。"""
    try:
        import cairosvg
        from PIL import Image
        import io
        png_bytes = cairosvg.svg2png(
            url=str(svg_path),
            output_width=size,
            output_height=size,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        # 白底（SVG 背景通常透明）
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert("RGB")
    except ImportError:
        raise SystemExit(
            "請先安裝 cairosvg：pip install cairosvg pillow\n"
            "若出現 libcairo 錯誤：sudo apt install libcairo2"
        )

# ── 主程式 ────────────────────────────────────────────
def main():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("請先安裝 pillow：pip install pillow")

    canvas = Image.new("RGB", (IMG_W, IMG_H), color=(240, 240, 240))
    draw   = ImageDraw.Draw(canvas)

    # 嘗試載入系統字型（顯示中文標籤用）
    cjk_fonts = [
        "/mnt/c/Windows/Fonts/msjh.ttc",        # 微軟正黑
        "/mnt/c/Windows/Fonts/kaiu.ttf",         # 標楷體
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    label_font = None
    for fp in cjk_fonts:
        try:
            label_font = ImageFont.truetype(fp, 16)
            break
        except Exception:
            pass
    if label_font is None:
        label_font = ImageFont.load_default()
        print("⚠ 找不到 CJK 字型，標籤可能顯示方塊；可用 msjh.ttc 解決")

    # ── 畫行標籤（字符名稱）──────────────────────────
    for ci, ch in enumerate(COLS):
        x = LABEL_W + ci * (CELL_W + PADDING) + CELL_W // 2
        y = HEADER_H // 2
        draw.text((x, y), ch, fill=(30, 30, 30), font=label_font, anchor="mm")

    # ── 逐格處理 ─────────────────────────────────────
    missing = []
    for ri, (font_name, prompt) in enumerate(ROWS):
        # 列標籤
        short = next(s for fn, s in FONTS if fn == font_name)
        row_label = f"{short}\n{prompt}"
        y_label = HEADER_H + ri * (CELL_H + PADDING) + CELL_H // 2
        draw.text((LABEL_W // 2, y_label), row_label,
                  fill=(30, 30, 30), font=label_font, anchor="mm")

        for ci, ch in enumerate(COLS):
            svg_name = f"{font_name}_{prompt}_{ch}.svg"
            svg_path = SVG_DIR / svg_name
            x = LABEL_W + ci * (CELL_W + PADDING)
            y = HEADER_H + ri * (CELL_H + PADDING)

            if not svg_path.exists():
                # 格子標紅，標記缺失
                draw.rectangle([x, y, x+CELL_W, y+CELL_H], fill=(255, 200, 200))
                draw.text((x + CELL_W//2, y + CELL_H//2), "?",
                          fill=(200, 0, 0), font=label_font, anchor="mm")
                missing.append(svg_name)
                continue

            try:
                cell_img = svg_to_pil(svg_path, CELL_W)
                canvas.paste(cell_img, (x, y))
            except Exception as e:
                draw.rectangle([x, y, x+CELL_W, y+CELL_H], fill=(255, 220, 180))
                draw.text((x + CELL_W//2, y + CELL_H//2), "err",
                          fill=(180, 80, 0), font=label_font, anchor="mm")
                print(f"  ✗ 渲染失敗：{svg_name}  ({e})")

        print(f"  列 {ri+1}/6 完成：{short} {prompt}")

    # ── 存檔 ─────────────────────────────────────────
    canvas.save(OUT_FILE)
    print(f"\n✅ 已輸出：{OUT_FILE}  ({IMG_W}×{IMG_H} px)")
    if missing:
        print(f"⚠ 缺少 {len(missing)} 個 SVG：{missing}")

if __name__ == "__main__":
    main()
