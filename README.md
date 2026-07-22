# 基於視覺語言模型之手寫字形向量圖編輯
# Vision-Language Models for Handwriting Glyph Vector Graphics Editing
<br>
<div align="center">
    <img src="media" width="100%">
</div>
<br><br>


## Abstract
Prompt-driven font editing is an emerging application of vision-language mod-
els in graphics. The recent FontCLIP couples cross-modal semantic alignment
with differentiable rasterization to optimize a glyph’s Bézier control points at in-
ference time, editing it into attribute-specific styles such as thin or bold from a
prompt. Such methods, however, have been validated mainly on Latin computer
typefaces; Chinese characters appear only as a single illustrative glyph, with neither
systematic treatment nor quantitative evaluation, leaving Chinese handwriting edit-
ing—structurally more complex and carrying personal style—unexplored. Using
FontCLIP as its base framework, this study presents a systematic evaluation and
optimization of VLM-based Chinese handwriting editing. Twenty representative
characters were optimized on a handwriting font against a thin computer typeface
under three prompts for 120 trials, with a systematic loss-weight search yielding a
reproducible hyperparameter configuration. To address the original method’s lack
of quantitative evaluation, we adopt the direction cosine, a metric aligned with Font-
CLIP’s training direction loss, to measure how well a glyph’s movement in seman-
tic space aligns with the target direction. Across all combinations, glyphs consis-
tently moved toward the intended direction: thin and bold were stable, whereas
the more abstract relaxed prompt was markedly weaker, reflecting a cross-cultural
gap in FontCLIP’s Western-dominated training data. Closed contours, rather than
stroke count alone, were the key factor limiting quality—open-contour characters
deformed stably while closed-contour ones collapsed under thin optimization—and
semantic alignment and visual legibility proved to be two independent dimensions.
Overall, the adopted metric, the hyperparameter configuration established here, and
the ability-boundary analysis fill this gap for Chinese handwriting editing.

## Approach
Our work builds upon **FontCLIP** by extending its capabilities to **handwritten fonts**. 

The original FontCLIP is a [CLIP](https://github.com/openai/CLIP) model fine-tuned with [a font dataset](https://www.dgp.toronto.edu/~donovan/font/), which explored several fine-tuning approaches integrated into a Python class named `ExCLIP`. 

For the foundational framework, documentation, and the original implementation, please refer to the [official FontCLIP repository](https://github.com/yukistavailable/FontCLIP).

## Applications

Based on our extended framework, we focus on the following application tailored for handwritten text:

* **Text-Guided Attribute Editing**: We demonstrate the ability to manipulate specific stylistic attributes of handwritten fonts using natural language prompts. By providing descriptive prompts such as "thin," "bold," or "italic," the model can dynamically adjust the vector outlines of handwritten characters while preserving their original authentic writing style.



### Vector optimization
<br>
<div align="center">
    <img src="media/FontCLIP_pip" width="100%">
</div>
<br><br>




## Setup
### Create environment
1. For finetuning CLIP
```bash
conda create -y --name fontclip python=3.8.15
conda activate fontclip
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tqdm ftfy regex gdown
pip install gradio==3.40.0
pip install httpx==0.24.1
```


2. For vector optimization
```
conda install -y numpy scikit-image
conda install -y -c anaconda cmake=3.22.1
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom freetype-py shapely ttf save_svg
pip install opencv-python==4.5.4.60  
pip install kornia==0.6.8
pip install wandb
pip install shapely

# install diffvg
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

Please be careful that the version of each library is suitable for diffvg. - see [the issue](https://github.com/BachiLi/diffvg/issues/37#issuecomment-1336335574) for details
### Download and setup a font dataset
```bash
python setup_data.py
```

## Finetune CLIP to produce FontCLIP
You can run the processor finetuning using the following command.

__:warning: gwfonts from [the download website](https://www.dgp.toronto.edu/~donovan/font/) lacks more than 50 font files necessary for running the finetuning program. Download them manually. (see https://github.com/yukistavailable/FontCLIP/issues/3)__
```
python train.py --random_prompt_num_per_font 10000 --sample_num 50 --color_jitter_sample_num 200 --use_lora_text
```

## ExCLIP
We have tried several finetuning methods (direct finetuning, [CoOp](https://github.com/KaiyangZhou/CoOp/), [VPT](https://github.com/KMnP/vpt), [LoRA](https://github.com/microsoft/LoRA), and [OFT](https://github.com/Zeju1997/oft)) and integrate them into one Python class named `ExCLIP`. - see [ex_clip.py](models/ex_clip.py) for details

## Licenses
### Main License
This project, based on [CLIP](https://github.com/openai/CLIP/), is licensed under MIT License - see the [LICENSE_MIT](LICENSE_MIT.md) for details

### Additional Licenses
The source codes for [CoOp](https://github.com/KaiyangZhou/CoOp/) in ExCLIP, is licensed under MIT License - see the [LICENSE_MIT](LICENSE_MIT.md) for details

The source codes for [VPT](https://github.com/KMnP/vpt) in ExCLIP, is licensed under CC-BY-NC 4.0 License - see the [LICENSE.CC_BY_NC_SA_4.0](LICENSE.CC_BY_NC_SA_4.0.md) for details

The source codes for [OFT](https://github.com/Zeju1997/oft) in ExCLIP, is licensed under MIT License - see the [LICENSE_MIT](LICENSE_MIT.md) for details

The source codes for vector optimization are based on [Word-As-Image](https://github.com/Shiriluz/Word-As-Image), particularly the files under `optimizer` folder. - see the [LICENSE.CC_BY_NC_SA_4.0](LICENSE.CC_BY_NC_SA_4.0.md) for details


## Attribution
The source codes for vector optimization are based on [Word-As-Image](https://github.com/Shiriluz/Word-As-Image) created by Shiriluz.
The original work can be found at https://github.com/Shiriluz/Word-As-Image and is licensed under CC BY-NC-SA 4.0.
