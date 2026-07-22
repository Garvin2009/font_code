#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ctrlpt_morph.py — 控制點 morph 影片 + 單張控制點圖 產生器
（本機在 WSL ~/FontCLIP_mine/ 執行）

輸入＝FontCLIP 每 iter 的向量快照資料夾 video-svg/（run_test_complex.py 開了
cfg.save.video 後會產生：output_ctrlpt_demo/<字型>/conf5_<prompt>_<字>/<字型>/<sig>/video-svg/iter0000.svg ...）。

三種輸出（同一份 video-svg 可一次全出）：
  1) dots  ── 透明背景、只有紅色「結構點」在動         → <out>_dots.gif（透明）＋ <out>_dots.mp4（白底備用）
  2) glyph ── 黑色粗體字形在變（不畫點）               → <out>_glyph.gif ＋ <out>_glyph.mp4
  3) still ── 單一張控制點圖（淡字形輪廓＋紅點），做簡報頁 7「學 vs A」用 → <out>.png（透明）

安裝：pip install cairosvg pillow imageio numpy   （要 MP4 再 pip install "imageio[ffmpeg]"）

用法：
  # 兩支影片（dots + glyph）一次出：
  python3 make_ctrlpt_morph.py \
      --svgdir output_ctrlpt_demo/CircleFont_v2/conf5_thin_學/CircleFont_v2/<sig>/video-svg \
      --out 資工所論文/fig/morph_學_thin --mode dots,glyph --fps 12 --dot 4

  # 只出 dots（透明結構點影片）：
  python3 make_ctrlpt_morph.py --svgdir <.../video-svg> --out fig/morph_學_thin --mode dots

  # 頁 7 用：學「原始字形」的單張控制點圖（用 svg-init/init.svg 最乾淨）：
  python3 make_ctrlpt_morph.py --still \
      --svgfile output_ctrlpt_demo/CircleFont_v2/conf5_thin_學/CircleFont_v2/<sig>/svg-init/init.svg \
      --out 資工所論文/fig/ctrlpts_學 --dot 5

參數：
  --svgdir       每 iter SVG 快照資料夾（做影片用；會自動抓 *.svg 依 iter 排序）
  --svgfile      單一 SVG（做 --still 用；沒給就抓 --svgdir 最後一張）
  --out          輸出檔名（不含副檔名）
  --mode         影片模式，逗號分隔：dots,glyph（預設）；也可只給其一
  --still        改成輸出「單張控制點圖」而非影片
  --no-glyph     （--still 時）不畫淡字形、只留純紅點
  --glyph-alpha  （--still 時）字形輪廓淡化程度 0~1，預設 0.28
  --size         渲染邊長 px（預設 480）
  --fps          影片每秒張數（預設 12）
  --dot          控制點半徑 px（預設 4；設 0＝不畫點）
  --hold         影片最後一張多停幾格（預設 8）
  --step         每隔幾張取一張加速（預設 1）
"""
import argparse, glob, os, re, sys, io


def natural_key(pth):
    nums = re.findall(r'\d+', os.path.basename(pth))
    return (int(nums[-1]) if nums else 0, pth)


def parse_points(svg_text):
    """從 SVG 所有 path 的 d 取出座標點（含 bezier 控制點）。純數字對解析，夠用來畫結構點。"""
    pts = []
    for d in re.findall(r'\bd\s*=\s*"([^"]+)"', svg_text):
        nums = re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', d)
        for i in range(0, len(nums) - 1, 2):
            try:
                pts.append((float(nums[i]), float(nums[i + 1])))
            except ValueError:
                pass
    return pts


def get_viewbox(svg_text):
    m = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_text)
    if m:
        v = [float(x) for x in m.group(1).replace(',', ' ').split()]
        if len(v) == 4:
            return v

    def dim(name):
        mm = re.search(name + r'\s*=\s*"?([\d.]+)', svg_text)
        return float(mm.group(1)) if mm else 512.0

    return [0, 0, dim('width'), dim('height')]


def _read(path):
    return open(path, 'r', encoding='utf-8', errors='ignore').read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--svgdir')
    ap.add_argument('--svgfile')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='dots,glyph')
    ap.add_argument('--still', action='store_true')
    ap.add_argument('--no-glyph', dest='no_glyph', action='store_true')
    ap.add_argument('--glyph-alpha', dest='glyph_alpha', type=float, default=0.28)
    ap.add_argument('--size', type=int, default=480)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--dot', type=int, default=4)
    ap.add_argument('--hold', type=int, default=8)
    ap.add_argument('--step', type=int, default=1)
    args = ap.parse_args()

    try:
        import cairosvg
        from PIL import Image, ImageDraw
        import imageio.v2 as imageio
    except ImportError as e:
        sys.exit('缺套件：%s\n請先 pip install cairosvg pillow imageio numpy' % e)

    DOT_FILL = (220, 40, 40)
    DOT_LINE = (120, 0, 0)

    # ── 小工具 ──────────────────────────────────────────
    def render_glyph(svg_text, transparent=True):
        """把一張 SVG 渲染成 RGBA。transparent=True → 背景透明（只有黑色字形）。"""
        bg = None if transparent else 'white'
        png = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'),
                               output_width=args.size, output_height=args.size,
                               background_color=bg)
        return Image.open(io.BytesIO(png)).convert('RGBA')

    def draw_dots(img, svg_text):
        if args.dot <= 0:
            return
        vx, vy, vw, vh = get_viewbox(svg_text)
        sx = args.size / vw if vw else 1
        sy = args.size / vh if vh else 1
        d = ImageDraw.Draw(img)
        r = args.dot
        for (px, py) in parse_points(svg_text):
            cx = (px - vx) * sx
            cy = (py - vy) * sy
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=DOT_FILL + (255,), outline=DOT_LINE + (255,))

    def frame_dots(svg_text):
        """透明畫布，只有紅點。"""
        canvas = Image.new('RGBA', (args.size, args.size), (0, 0, 0, 0))
        draw_dots(canvas, svg_text)
        return canvas

    def frame_glyph(svg_text):
        """白底黑字，不畫點。"""
        canvas = Image.new('RGBA', (args.size, args.size), (255, 255, 255, 255))
        canvas.alpha_composite(render_glyph(svg_text, transparent=True))
        return canvas

    def on_white(rgba):
        bgim = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        bgim.alpha_composite(rgba)
        return bgim.convert('RGB')

    def ensure_dir(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    def save_transparent_gif(frames_rgba, path):
        frames = frames_rgba + [frames_rgba[-1]] * max(0, args.hold)

        def to_p(im):
            alpha = im.getchannel('A')
            p = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
            mask = alpha.point(lambda v: 255 if v <= 128 else 0)
            p.paste(255, mask=mask)   # index 255 = 透明
            return p

        ps = [to_p(f) for f in frames]
        ensure_dir(path)
        ps[0].save(path, save_all=True, append_images=ps[1:],
                   duration=int(1000 / args.fps), loop=0,
                   transparency=255, disposal=2, optimize=False)

    def save_opaque(frames_rgba, prefix):
        frames = frames_rgba + [frames_rgba[-1]] * max(0, args.hold)
        rgb = [on_white(f) for f in frames]
        import numpy as np
        gif = prefix + '.gif'
        ensure_dir(gif)
        imageio.mimsave(gif, [np.array(f) for f in rgb], duration=1.0 / args.fps, loop=0)
        print('✓ GIF:', gif)
        try:
            mp4 = prefix + '.mp4'
            imageio.mimsave(mp4, [np.array(f) for f in rgb], fps=args.fps)
            print('✓ MP4:', mp4)
        except Exception as e:
            print('（MP4 略過，需 pip install "imageio[ffmpeg]"）：', e)

    # ── 模式 A：單張控制點圖 ─────────────────────────────
    if args.still:
        svgfile = args.svgfile
        if not svgfile and args.svgdir:
            cand = sorted(glob.glob(os.path.join(args.svgdir, '*.svg')), key=natural_key)
            svgfile = cand[-1] if cand else None
        if not svgfile or not os.path.exists(svgfile):
            sys.exit('--still 需要 --svgfile（或可從 --svgdir 抓到 .svg）；找不到 SVG')

        svg_text = _read(svgfile)
        canvas = Image.new('RGBA', (args.size, args.size), (0, 0, 0, 0))
        if not args.no_glyph:
            g = render_glyph(svg_text, transparent=True)
            a = g.getchannel('A').point(lambda v: int(v * args.glyph_alpha))
            g.putalpha(a)                      # 字形輪廓淡化當背景
            canvas.alpha_composite(g)
        draw_dots(canvas, svg_text)
        out = args.out + '.png'
        ensure_dir(out)
        canvas.save(out)
        print('✓ 單張控制點圖（透明 PNG）:', out)
        print('  來源 SVG:', svgfile)
        return

    # ── 模式 B：影片 ────────────────────────────────────
    if not args.svgdir:
        sys.exit('影片模式需要 --svgdir（指向 video-svg 資料夾）')
    svgs = sorted(glob.glob(os.path.join(args.svgdir, '*.svg')), key=natural_key)
    svgs = svgs[::max(1, args.step)]
    if not svgs:
        sys.exit('在 %s 找不到任何 .svg（確認 cfg.save.video=True 有跑、路徑指到 video-svg）' % args.svgdir)
    print('找到 %d 張快照，渲染中…' % len(svgs))

    modes = [m.strip() for m in args.mode.split(',') if m.strip()]
    texts = [_read(p) for p in svgs]

    if 'dots' in modes:
        print('→ dots（透明結構點）…')
        frames = []
        for k, t in enumerate(texts):
            frames.append(frame_dots(t))
            if (k + 1) % 40 == 0:
                print('   %d/%d' % (k + 1, len(texts)))
        save_transparent_gif(frames, args.out + '_dots.gif')
        print('✓ 透明 GIF:', args.out + '_dots.gif')
        save_opaque(frames, args.out + '_dots')   # 白底 mp4 備用（mp4 無法透明）

    if 'glyph' in modes:
        print('→ glyph（黑色粗體字形）…')
        frames = []
        for k, t in enumerate(texts):
            frames.append(frame_glyph(t))
            if (k + 1) % 40 == 0:
                print('   %d/%d' % (k + 1, len(texts)))
        save_opaque(frames, args.out + '_glyph')

    if 'overlay' in modes:   # 保留舊行為：字形＋點疊一起
        print('→ overlay（字形＋點）…')
        frames = []
        for t in texts:
            f = frame_glyph(t)
            draw_dots(f, t)
            frames.append(f)
        save_opaque(frames, args.out + '_overlay')

    print('\n放進簡報：PowerPoint > 插入 > 影片/GIF；dots 的透明 GIF 可直接疊在字形影片上。')


if __name__ == '__main__':
    main()
