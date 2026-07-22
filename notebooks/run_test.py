#!/usr/bin/env python
# coding: utf-8
import multiprocessing
import os
import csv
from datetime import datetime

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import faulthandler
faulthandler.enable()

from IPython.display import SVG, display
import yaml
from easydict import EasyDict as edict
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageFont
import sys
sys.path.append('../')
from optimizer.optimize import (
    Config,
    draw_image_through_svg_from_font_path,
    get_signature,
    train,
    create_svg_from_font,
    show_outline,
    change_size_transform,
    device,
    draw_single_character,
)
from optimizer.util_tools import update
import tqdm
import tqdm.auto
import tqdm.notebook

tqdm.notebook.tqdm = tqdm.std.tqdm
tqdm.auto.tqdm = tqdm.std.tqdm

# ── 基本設定 ──────────────────────────────────────────
USE_WANDB = 0
WANDB_USER = ''

checkpoint_path = '../model_checkpoints/ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug50_cj200_lbound_of_scale0.35_max_attr_num_3_random_p_num_10000_geta0.2_use_negative_lr0.0002-0.1_image_file_dir.pt'

# ── 實驗字清單 ────────────────────────────────────────
# 正式實驗 20 字（與論文 3.5 節一致）。若只想先試跑驗證，改成 ["一"] 之類單字。
optimized_letters = [
    "一","二","三","十","人","大","小","山","川","土",
    "木","水","火","日","月","口","田","力","又","子"
]

# ── 兩個字體設定 ──────────────────────────────────────
font_configs = [ 
    {
        "font_path": "../gwfonts/NotoSansJP-ExtraLight.ttf",
        "visual_font_path": "../gwfonts/NotoSansJP-ExtraLight.ttf",
    },
    {
        "font_path": "../gwfonts/CircleFont_v2.ttf",
        "visual_font_path": "../gwfonts/CircleFont_v2.ttf",
    },
   
]

NEW_OUTPUT_BASE = "../output_v3_direction"   # v3：direction loss 正式實驗，與舊 output 分開保存


# ─────────────────────────────────────────────────────
def get_config(
    optimized_letter,
    font_path,
    visual_font_path=None,
    num_iter=201,
    use_lr_scheduler=True,
    target_image_path=None,
    semantic_concept=None,
    do_preprocess=True,
    lr=1.0,
    use_aug=False,
    image_file_path=None,
    use_L2_loss=False,
    L2_loss_w=0.5,
    use_fclip=False,
    fclip_loss_w=1.0,
    use_fclip_direction_loss=False,
    use_fclip_direction_loss_vision=False,
    use_fclip_direction_loss_only=False,
    ref_semantic_concept=None,
    ref_image_file_path=None,
    fclip_direction_loss_w=1.0,
    checkpoint_path=None,
    use_conformal_loss=False,
    conformal_loss_w=1.0,
    use_tone_loss=False,
    tone_loss_w=5.0,
    use_G1_loss=False,
    G1_loss_w=5.0,
    skip_corner_threshold=-0.85,
    skip_corners=True,
    use_laplacian_loss=False,
    laplacian_loss_w=0.5,
    only_edge_laplacian=True,
    use_laplacian_between_beziers_loss=False,
    laplacian_between_beziers_loss_w=1.0,
    laplacian_between_beziers_loss_threshold=-0.8,
    use_Xing_loss=False,
    Xing_loss_w=1.0,
    use_direction_loss=False,
    direction_loss_w=1.0,
):
    visual_optimize = True
    if semantic_concept is not None:
        visual_optimize = False

    char_size = 150
    size = 200
    multiple_attributes = False
    multiple_attributes_preserve_init = False
    target_attributes = ["thin", "formal", "legible"]
    reduce_cp = False
    epsilon = 50
    target_attributes_weights = [fclip_loss_w] + [
        fclip_loss_w / 10 for _ in range(len(target_attributes) - 1)
    ]
    use_cos_loss = False
    cos_loss_w = 0.05
    num_per_curve = 8
    skip_control_points = True
    skip_edge_laplacian = False
    skip_edge_cos = True

    cfg = Config(
        font_path, optimized_letter, None, optimized_letter,
        use_wandb=USE_WANDB, wandb_user=WANDB_USER,
        num_iter=num_iter, char_size=char_size, size=size,
        do_preprocess=do_preprocess, lr=lr, use_aug=use_aug,
        use_visual_encoder=False,
        fclip_loss=use_fclip, fclip_loss_w=fclip_loss_w,
        use_fclip_direction_loss=use_fclip_direction_loss,
        use_fclip_direction_loss_vision=use_fclip_direction_loss_vision,
        use_fclip_direction_loss_only=use_fclip_direction_loss_only,
        ref_semantic_concept=ref_semantic_concept,
        ref_image_file_path=ref_image_file_path,
        fclip_direction_loss_w=fclip_direction_loss_w,
        laplacian_loss_w=laplacian_loss_w,
        cos_loss_w=cos_loss_w, G1_loss_w=G1_loss_w, L2_loss_w=L2_loss_w,
        conformal_loss_w=conformal_loss_w, tone_loss_w=tone_loss_w,
        multiple_attributes=multiple_attributes,
        multiple_text_encoders=multiple_attributes_preserve_init,
        target_attributes=target_attributes,
        target_attributes_weights=target_attributes_weights,
        use_tone_loss=use_tone_loss,
        use_conformal_loss=use_conformal_loss,
        use_laplacian_loss=use_laplacian_loss,
        use_cos_loss=use_cos_loss, use_G1_loss=use_G1_loss,
        reduce_cp=reduce_cp, epsilon=epsilon,
        checkpoint_path=checkpoint_path,
        visual_optimize=visual_optimize,
        image_file_path=image_file_path,
        use_tone_loss_schedular=False,
        use_lr_scheduler=use_lr_scheduler,
        use_L2_loss=use_L2_loss,
        is_counter=False,
        num_per_curve=num_per_curve,
        skip_control_points=skip_control_points,
        skip_corners=skip_corners,
        skip_corner_threshold=skip_corner_threshold,
        skip_edge_laplacian=skip_edge_laplacian,
        skip_edge_cos=skip_edge_cos,
        only_edge_laplacian=only_edge_laplacian,
        use_laplacian_between_beziers_loss=use_laplacian_between_beziers_loss,
        laplacian_between_beziers_loss_w=laplacian_between_beziers_loss_w,
        laplacian_between_beziers_loss_threshold=laplacian_between_beziers_loss_threshold,
        use_Xing_loss=use_Xing_loss, Xing_loss_w=Xing_loss_w,
        use_direction_loss=use_direction_loss, direction_loss_w=direction_loss_w,
    )

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
    del cfgs

    if target_image_path is not None:
        target_img = to_tensor(Image.open(target_image_path).convert("L").convert("RGB"))
        target_img = change_size_transform(size)(target_img).to(device)
    else:
        if image_file_path is not None:
            target_img = to_tensor(Image.open(image_file_path))
        else:
            target_img = draw_image_through_svg_from_font_path(
                visual_font_path, optimized_letter,
                svg_size=size, char_size=char_size, image_size=size,
            )
    cfg.target_img = target_img

    if semantic_concept is not None:
        print("================== semantic concept ====================")
        print(semantic_concept)
        print("========================================================")
    cfg.semantic_concept = semantic_concept
    return cfg

prompt_short_map = {
                # 論文正式實驗只用以下三個 prompt（italic 已移除）
                "thin, light handwriting": "thin",
                "bold, heavy handwriting": "bold",
                "relaxed, casual handwriting": "relax",
                }
# ─────────────────────────────────────────────────────
def process_single_char(optimized_letter, semantic_concept, param_set, font_path, visual_font_path):
    try:
        lr               = param_set["lr"]
        fclip_loss_w     = param_set["fclip_loss_w"]
        conformal_loss_w = param_set["conformal_loss_w"]
        num_iter         = 201
        target_iters     = [0, 50, 100, 150, 200]

        cfg = get_config(
            optimized_letter, font_path, visual_font_path, num_iter,
            use_lr_scheduler=True, lr=lr, use_aug=True,
            do_preprocess=True, semantic_concept=semantic_concept,
            image_file_path=None, target_image_path=None,
            use_L2_loss=False, L2_loss_w=0.5,
            checkpoint_path=checkpoint_path,
            use_fclip=True, fclip_loss_w=fclip_loss_w,
            # ── [v3 改動] 真正開啟 direction loss，方法與評估一致 ──
            # 目標方向 = semantic_concept(thin/bold/relax) - ref(formal,legible)
            use_fclip_direction_loss=True,
            ref_semantic_concept="formal, legible font",
            fclip_direction_loss_w=fclip_loss_w,   # 方向損失權重，沿用 fclip_loss_w=5.0
            use_conformal_loss=True, conformal_loss_w=conformal_loss_w,
            use_tone_loss=True, tone_loss_w=5.0,
            use_laplacian_loss=True, laplacian_loss_w=0.5,
            only_edge_laplacian=True,
            use_Xing_loss=False, use_direction_loss=False,
        )

        cfg.use_patch_based_loss = False
        cfg.use_global_disp_loss = False

        font_name    = os.path.splitext(os.path.basename(font_path))[0]
        prompt_short = prompt_short_map.get(semantic_concept, semantic_concept[:8])

        # 短資料夾名稱：output_final/CircleFont_v2/thin_一/
        cfg.log_dir = os.path.join(NEW_OUTPUT_BASE, font_name, f"{prompt_short}_{optimized_letter}")

        signature = get_signature(cfg, font_path, visual_font_path, None)
        print(f"\n🚀 [{font_name}][{optimized_letter}] prompt={prompt_short}")

        cfg.save.video = True
        cfg.save.image = True

        # ── train ──────────────────────────────────────
        result = train(cfg, signature)
        # v3：train 回傳 dict。向後相容舊版回傳 float 的情況。
        if isinstance(result, dict):
            final_dir_cos          = result.get("final_dir_cos", None)
            final_abs_cos          = result.get("final_abs_cos", None)
            final_clip_loss_legacy = result.get("final_clip_loss_legacy", None)
        else:
            final_dir_cos, final_abs_cos = None, None
            final_clip_loss_legacy = result

        # ── CSV ────────────────────────────────────────
        log_file    = os.path.join(NEW_OUTPUT_BASE, "clip_scores.csv")
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "時間","字體","字","prompt","fclip_w","conf_w","iter",
                    "final_dir_cos","final_abs_cos","final_clip_loss_legacy",
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                font_name, optimized_letter, semantic_concept,
                fclip_loss_w, conformal_loss_w, num_iter,
                final_dir_cos, final_abs_cos, final_clip_loss_legacy,
            ])

        # ── 動畫 SVG ───────────────────────────────────
        output_path = create_svg_from_font(
            target_iters, character=optimized_letter,
            font_path=font_path, output_path=None,
            signature=signature, target_attribute="",
            visual_optimize=True, visual_font_path=visual_font_path,
            char_size=150,
            base_path=os.path.join(NEW_OUTPUT_BASE, font_name, f"{prompt_short}_"),
        )
        print(f"✨ [{font_name}][{optimized_letter}] SVG: {output_path}")

    except Exception as e:
        print(f"❌ [{optimized_letter}] 錯誤: {e}")
        import traceback
        traceback.print_exc()

# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    
    
    param_sets = [
        {"lr": 1.0, "fclip_loss_w": 5.0, "conformal_loss_w": 5.0},
    ]

    # 兩個字體 × 20字 × 3 prompt = 120 個任務，約 8 小時
    tasks = [
        (char, prompt, params, fc["font_path"], fc["visual_font_path"])
        for fc in font_configs
        for char in optimized_letters
        for prompt in prompt_short_map
        for params in param_sets
    ]
    print(f"總共 {len(tasks)} 個任務，預計約 {len(tasks) * 4 // 60} 小時")

    crash_log_file = "crashes.txt"
    for idx, (char, prompt, params, fp, vfp) in enumerate(tasks):
        font_name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n--- [{idx+1}/{len(tasks)}] {font_name} | {char} | {prompt[:25]} ---")
        p = multiprocessing.Process(
            target=process_single_char,
            args=(char, prompt, params, fp, vfp)
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"💀 [{char}] 崩潰，跳過")
            with open(crash_log_file, "a", encoding="utf-8") as f:
                f.write(f"{font_name},{char},{prompt},{params}\n")

    print(f"\n🎉 全部完成！結果在 {NEW_OUTPUT_BASE}/")
    print(f"📊 CLIP 數據在 {NEW_OUTPUT_BASE}/clip_scores.csv")
