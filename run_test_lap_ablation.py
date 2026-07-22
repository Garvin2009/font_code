#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────────────────────
# 拉普拉斯平滑消融實驗（口試後補進論文用）
# 由 run_test4808_v3.py 改出，只改「是否開拉普拉斯」這一個變數，其餘完全相同。
#   對照組 A：laplacian_loss_w = 0.5（現行主實驗設定）
#   對照組 B：laplacian_loss_w = 0.0（關閉拉普拉斯）
# 字：木 水 日 月 學（簡單/封閉/複雜各涵蓋）；prompt：thin、bold（thin 最看得到鋸齒）
# 字型：只跑個人手寫字體 CircleFont_v2（主角）
# 規模：5 字 × 2 prompt × 2 條件 = 20 次，約 1.3 小時
# 產出：../output_lap_ablation/  ＋  clip_scores_lap.csv（多一欄 lap_w）
# 用法（在 ~/FontCLIP_mine/ 底下，跟 run_test4808_v3.py 同層執行）：
#   python run_test_lap_ablation.py
# ─────────────────────────────────────────────────────────────
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

checkpoint_path = './model_checkpoints/ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug50_cj200_lbound_of_scale0.35_max_attr_num_3_random_p_num_10000_geta0.2_use_negative_lr0.0002-0.1_image_file_dir.pt'

# ── 消融字清單（縮小版，只挑代表字）────────────────────
# 木/水=開放筆畫、日/月=封閉輪廓、學=高複雜度。若要更快，先改成 ["木"] 試跑。
optimized_letters = ["木", "水", "日", "月", "學"]

# ── 字型：只跑個人手寫字體（主角）──────────────────────
font_configs = [
    {
        "font_path": "./gwfonts/CircleFont_v2.ttf",
        "visual_font_path": "./gwfonts/CircleFont_v2.ttf",
    },
]

# ── 消融條件：拉普拉斯權重 開(0.5) vs 關(0.0) ───────────
LAP_W_VALUES = [0.5, 0.0]

NEW_OUTPUT_BASE = "./output_lap_ablation"   # 與主實驗 output_v3_direction 分開保存


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
                # 消融只用 thin / bold（thin 最容易看到鋸齒差異）
                "thin, light handwriting": "thin",
                "bold, heavy handwriting": "bold",
                }
# ─────────────────────────────────────────────────────
def process_single_char(optimized_letter, semantic_concept, param_set, font_path, visual_font_path):
    try:
        lr               = param_set["lr"]
        fclip_loss_w     = param_set["fclip_loss_w"]
        conformal_loss_w = param_set["conformal_loss_w"]
        lap_w            = param_set["lap_w"]          # ← [消融] 這次唯一變動的變數
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
            use_fclip_direction_loss=True,
            ref_semantic_concept="formal, legible font",
            fclip_direction_loss_w=fclip_loss_w,
            use_conformal_loss=True, conformal_loss_w=conformal_loss_w,
            use_tone_loss=True, tone_loss_w=5.0,
            # ── [消融核心] lap_w>0 才開拉普拉斯；lap_w=0 完全關閉 ──
            use_laplacian_loss=(lap_w > 0),
            laplacian_loss_w=lap_w,
            only_edge_laplacian=True,
            use_Xing_loss=False, use_direction_loss=False,
        )

        cfg.use_patch_based_loss = False
        cfg.use_global_disp_loss = False

        font_name    = os.path.splitext(os.path.basename(font_path))[0]
        prompt_short = prompt_short_map.get(semantic_concept, semantic_concept[:8])
        # 資料夾帶 lap 標籤，避免兩條件互相覆蓋：thin_木_lap0.5 / thin_木_lap0.0
        lap_tag = f"lap{lap_w}"
        run_tag = f"{prompt_short}_{lap_tag}"

        cfg.log_dir = os.path.join(NEW_OUTPUT_BASE, font_name, f"{run_tag}_{optimized_letter}")

        signature = get_signature(cfg, font_path, visual_font_path, None)
        print(f"\n🚀 [{font_name}][{optimized_letter}] prompt={prompt_short} lap_w={lap_w}")

        cfg.save.video = True
        cfg.save.image = True

        # ── train ──────────────────────────────────────
        result = train(cfg, signature)
        if isinstance(result, dict):
            final_dir_cos          = result.get("final_dir_cos", None)
            final_abs_cos          = result.get("final_abs_cos", None)
            final_clip_loss_legacy = result.get("final_clip_loss_legacy", None)
        else:
            final_dir_cos, final_abs_cos = None, None
            final_clip_loss_legacy = result

        # ── CSV（多一欄 lap_w）─────────────────────────
        log_file    = os.path.join(NEW_OUTPUT_BASE, "clip_scores_lap.csv")
        file_exists = os.path.isfile(log_file)
        os.makedirs(NEW_OUTPUT_BASE, exist_ok=True)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "時間","字體","字","prompt","lap_w","fclip_w","conf_w","iter",
                    "final_dir_cos","final_abs_cos","final_clip_loss_legacy",
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                font_name, optimized_letter, semantic_concept, lap_w,
                fclip_loss_w, conformal_loss_w, num_iter,
                final_dir_cos, final_abs_cos, final_clip_loss_legacy,
            ])

        # ── 動畫 SVG（before/after 網格素材）────────────
        output_path = create_svg_from_font(
            target_iters, character=optimized_letter,
            font_path=font_path, output_path=None,
            signature=signature, target_attribute="",
            visual_optimize=True, visual_font_path=visual_font_path,
            char_size=150,
            base_path=os.path.join(NEW_OUTPUT_BASE, font_name, f"{run_tag}_"),
        )
        print(f"✨ [{font_name}][{optimized_letter}] lap_w={lap_w} SVG: {output_path}")

    except Exception as e:
        print(f"❌ [{optimized_letter}] 錯誤: {e}")
        import traceback
        traceback.print_exc()

# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # 每個 lap_w 生一組 param_set（其餘超參數與主實驗完全相同）
    param_sets = [
        {"lr": 1.0, "fclip_loss_w": 5.0, "conformal_loss_w": 5.0, "lap_w": lap_w}
        for lap_w in LAP_W_VALUES
    ]

    # 1 字型 × 5 字 × 2 prompt × 2 條件 = 20 個任務
    tasks = [
        (char, prompt, params, fc["font_path"], fc["visual_font_path"])
        for fc in font_configs
        for char in optimized_letters
        for prompt in prompt_short_map
        for params in param_sets
    ]
    print(f"總共 {len(tasks)} 個任務，預計約 {len(tasks) * 4 // 60} 小時")

    crash_log_file = "crashes_lap.txt"
    for idx, (char, prompt, params, fp, vfp) in enumerate(tasks):
        font_name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n--- [{idx+1}/{len(tasks)}] {font_name} | {char} | {prompt[:25]} | lap_w={params['lap_w']} ---")
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
    print(f"📊 消融數據在 {NEW_OUTPUT_BASE}/clip_scores_lap.csv")
    print("   → 樞紐分析：同一(字,prompt)比 lap_w=0.5 vs 0.0 的 final_dir_cos，")
    print("     並用 output 裡的 SVG/PNG 做 before/after 視覺對照。")
