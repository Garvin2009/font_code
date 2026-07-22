"""
run_test_emotion.py
===================
情緒屬性探測實驗（動機：顏真卿祭姪文稿的情緒→字形映射）

目的：
  探測 FontCLIP 單字控制點優化能否表達「情緒類」高階屬性
  （angry / happy / sad），並驗證兩個推論：
    (1) 情緒屬性缺乏單一低階視覺軸，模型可能塌縮到最接近的
        低階屬性（例如 angry ≈ 變粗變糙，類似 bold）。
    (2) 書法情緒主要住在跨字/章法層級，單字優化在結構上到不了
        → domain mismatch（方向餘弦偏低、辨別力下降即為佐證）。

設計：
  - 8 個字（簡單開放 / 含封閉輪廓 / 高複雜度各取樣）
  - 3 個情緒 prompt：angry / happy / sad
  - 2 個字型：CircleFont_v2（個人手寫字體）、NotoSansJP-ExtraLight（對照）
  - conf_w 固定 5.0（主實驗確定超參數），其餘超參數與主實驗一致
  共 8 × 3 × 2 = 48 筆

輸出：
  - SVG / PNG 存至 output_emotion/
  - CSV：output_emotion/results.csv

備註：
  - ref prompt 沿用 "formal, legible font"，讓情緒的方向餘弦與主實驗
    thin/bold/relaxed 落在同一量尺、可直接對照。
  - 若想改用情緒中性基準，把 get_config 內 ref_semantic_concept 改成
    "neutral, calm handwriting" 即可（但會與主實驗數值不可直接比較）。
"""

import multiprocessing
import os
import sys
import csv
import traceback
from datetime import datetime

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import faulthandler
faulthandler.enable()

import yaml
from easydict import EasyDict as edict
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizer.optimize import (
    Config,
    draw_image_through_svg_from_font_path,
    get_signature,
    train,
    change_size_transform,
    device,
)
from optimizer.util_tools import update

# ── 基本設定 ──────────────────────────────────────────
USE_WANDB = 0
WANDB_USER = ''

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

checkpoint_path = os.path.join(SCRIPT_DIR,
    'model_checkpoints/ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug50_cj200_lbound_of_scale0.35_max_attr_num_3_random_p_num_10000_geta0.2_use_negative_lr0.0002-0.1_image_file_dir.pt'
)

# ── 字型路徑 ──────────────────────────────────────────
font_configs = [
    {
        "font_path": os.path.join(SCRIPT_DIR, "gwfonts/CircleFont_v2.ttf"),
        "visual_font_path": os.path.join(SCRIPT_DIR, "gwfonts/CircleFont_v2.ttf"),
    },
    {
        "font_path": os.path.join(SCRIPT_DIR, "gwfonts/NotoSansJP-ExtraLight.ttf"),
        "visual_font_path": os.path.join(SCRIPT_DIR, "gwfonts/NotoSansJP-ExtraLight.ttf"),
    },
]

# ── 取樣字（簡單開放 / 含封閉輪廓 / 高複雜度）──────────
EMOTION_CHARS = ["木"]

# ── 情緒 Prompt ───────────────────────────────────────
PROMPTS = {
    "angry": "angry, aggressive handwriting",
    "happy": "happy, cheerful handwriting",
    "sad":   "sad, sorrowful handwriting",
}

# ── conf_w 固定值（主實驗確定超參數）──────────────────
CONF_W = 5.0

# ── 輸出 ──────────────────────────────────────────────
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output_emotion")
CSV_PATH = os.path.join(OUTPUT_ROOT, "results.csv")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ── get_config（與主實驗相同邏輯，conf_w 固定 5.0）────────
def get_config(
    optimized_letter, font_path, visual_font_path,
    semantic_concept, conformal_loss_w,
    num_iter=201, lr=1.0,
):
    char_size = 150
    size = 200
    fclip_loss_w = 5.0

    cfg = Config(
        font_path, optimized_letter, None, optimized_letter,
        use_wandb=USE_WANDB, wandb_user=WANDB_USER,
        num_iter=num_iter, char_size=char_size, size=size,
        do_preprocess=True, lr=lr, use_aug=True,
        use_visual_encoder=False,
        fclip_loss=True, fclip_loss_w=fclip_loss_w,
        use_fclip_direction_loss=True,
        ref_semantic_concept="formal, legible font",
        fclip_direction_loss_w=fclip_loss_w,
        conformal_loss_w=conformal_loss_w,
        tone_loss_w=5.0,
        laplacian_loss_w=0.5,
        use_tone_loss=True,
        use_conformal_loss=True,
        use_laplacian_loss=True,
        use_cos_loss=False,
        use_G1_loss=False,
        use_L2_loss=False,
        use_Xing_loss=False,
        use_direction_loss=False,
        only_edge_laplacian=True,
        skip_control_points=True,
        skip_corners=True,
        skip_corner_threshold=-0.85,
        skip_edge_laplacian=False,
        skip_edge_cos=True,
        use_tone_loss_schedular=False,
        use_lr_scheduler=True,
        is_counter=False,
        num_per_curve=8,
        checkpoint_path=checkpoint_path,
        visual_optimize=False,   # semantic_concept 模式
        multiple_attributes=False,
        multiple_text_encoders=False,
        target_attributes=["thin", "formal", "legible"],
        target_attributes_weights=[fclip_loss_w, fclip_loss_w/10, fclip_loss_w/10],
    )

    # ── YAML 合併（get_signature 和 train 需要 cfg.loss 等 YAML 欄位）──
    with open(cfg.config, "r") as f:
        cfg_full = yaml.load(f, Loader=yaml.Loader)

    cfg_key = cfg.experiment
    cfgs = [vars(cfg)]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get("parent_config", "baseline")
    tmp_cfg = edict()
    for options in reversed(cfgs):
        update(tmp_cfg, options)
    cfg = tmp_cfg

    # ── target image ──
    target_img = draw_image_through_svg_from_font_path(
        visual_font_path, optimized_letter,
        svg_size=size, char_size=char_size, image_size=size,
    )
    cfg.target_img = target_img
    cfg.semantic_concept = semantic_concept
    return cfg


def process_single(optimized_letter, semantic_concept, prompt_key, font_path, visual_font_path, conf_w):
    try:
        font_name = os.path.splitext(os.path.basename(font_path))[0]

        cfg = get_config(
            optimized_letter, font_path, visual_font_path,
            semantic_concept, conformal_loss_w=float(conf_w),
        )

        cfg.use_patch_based_loss = False
        cfg.use_global_disp_loss = False
        cfg.log_dir = os.path.join(OUTPUT_ROOT, font_name, f"{prompt_key}_{optimized_letter}")
        cfg.save.video = True
        cfg.save.image = True

        signature = get_signature(cfg, font_path, visual_font_path, None)
        print(f"\n🚀 [{font_name}][{optimized_letter}] emotion={prompt_key}")

        result = train(cfg, signature)

        if isinstance(result, dict):
            final_dir_cos          = result.get("final_dir_cos")
            final_abs_cos          = result.get("final_abs_cos")
            final_clip_loss_legacy = result.get("final_clip_loss_legacy")
        else:
            final_dir_cos, final_abs_cos, final_clip_loss_legacy = None, None, result

        print(f"  ✓ dir_cos={final_dir_cos}  abs_cos={final_abs_cos}")

        file_exists = os.path.isfile(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["時間", "font", "char", "emotion",
                                  "final_dir_cos", "final_abs_cos", "final_clip_loss_legacy"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                font_name, optimized_letter, prompt_key,
                final_dir_cos, final_abs_cos, final_clip_loss_legacy,
            ])

    except Exception as e:
        print(f"❌ [{optimized_letter}] emotion={prompt_key} 錯誤: {e}")
        traceback.print_exc()
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.path.splitext(os.path.basename(font_path))[0],
                optimized_letter, prompt_key,
                None, None, None,
            ])


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # ── 字型路徑確認 ──
    print("字型路徑確認：")
    for fc in font_configs:
        status = "✓" if os.path.exists(fc["font_path"]) else "✗ 找不到"
        print(f"  {status}  {fc['font_path']}")

    tasks = [
        (char, prompt_text, prompt_key, fc["font_path"], fc["visual_font_path"], CONF_W)
        for fc in font_configs
        for char in EMOTION_CHARS
        for prompt_key, prompt_text in PROMPTS.items()
    ]
    print(f"\n總共 {len(tasks)} 個任務（{len(EMOTION_CHARS)} 字 × {len(PROMPTS)} 情緒 × {len(font_configs)} 字型）\n")

    for idx, (char, prompt_text, prompt_key, fp, vfp, conf_w) in enumerate(tasks):
        font_name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n--- [{idx+1}/{len(tasks)}] {font_name} | {char} | {prompt_key} ---")
        p = multiprocessing.Process(
            target=process_single,
            args=(char, prompt_text, prompt_key, fp, vfp, conf_w)
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"💀 [{char}] emotion={prompt_key} 崩潰，跳過")

    print(f"\n🎉 完成！結果在 {CSV_PATH}")
