# ControlCom-Image-Composition

This is the official repository for the following research paper:

> **ControlCom: Controllable Image Composition using Diffusion Model**  [[arXiv]](https://arxiv.org/pdf/2308.10040.pdf)<br>
>
> Bo Zhang, Yuxuan Duan, Jun Lan, Yan Hong, Huijia Zhu, Weiqiang Wang, Li Niu<br>

**Part of our ControlCom has been integrated into our image composition toolbox libcom https://github.com/bcmi/libcom. Welcome to visit and try ＼(^▽^)／** 

## Table of Contents
+ [Demo](#Demo)
+ [Task Definition](#Task-definition)
+ [Network Architecture](#Network-architecture)
+ [FOSCom Dataset](#FOSCom-Dataset)
+ [Code and Model](#Code-and-model)
+ [Experiments](#Experiments)
+ [Evaluation](#Evaluation)


## Demo

The online demo of image composition can be found [here](https://bcmi.sjtu.edu.cn/home/niuli/demo_image_composition/).

## Task Definition


**In our controllable image composition model, we unify four tasks in one model using an 2-dim binary indicator vector, in which the first (*resp.*, second) dimension represents whether adjusting the foreground illumination (*resp.*, pose) to be compatible with background.**  1 means making adjustment and 0 means remaining the same. Therefore, (0,0) corresponds to image blending, (1,0) corresponds to image harmonization, (0,1) corresponds to view synthesis, (1,1) corresponds to generative composition. 

<p align='center'>  
  <img src='./figures/task.png'  width=70% />
</p>

Our method can selectively adjust partial foreground attributes. **Previous methods may adjust the foreground color/pose unexpectedly and even unreasonably, even when the foreground illumination and pose are already compatible with the background.** In the left part, the foreground pose is already compatible with background and previous methods make unnecessary adjustment. In the right part, the foreground illumination is already compatible with the background and previous methods adjust the foreground color in an undesirable manner. 

<p align='center'>  
  <img src='./figures/controllability_necessity.jpg'  width=90% />
</p>

**The (0,0), (1,0) versions without changing foreground pose are very robust and generally well-behaved, but some tiny details may be lost or altered. The (0,1), (1,1) versions changing foreground pose are less robust and may produce the results with distorted structures or noticeable artifacts.** For foreground pose variation, we recommend more robust [ObjectStitch](https://github.com/bcmi/ObjectStitch-Image-Composition) and [MureObjectStitch](https://github.com/bcmi/MureObjectStitch-Image-Composition).

**Note that in the provided foreground image, the foreground object's length and width should fully extend to the edges of the image (see our example), otherwise the performance would be severely affected.**

## Network Architecture

Our method is built upon stable diffusion and the network architecture is shown as follows.

<p align='center'>  
  <img src='./figures/architecture.png'  width=90% />
</p>

## FOSCom Dataset

- **Download link**:
  - [Dropbox](https://www.dropbox.com/scl/fi/c3ynuw7sya1r6f2khm828/FOSCom.zip?rlkey=xif0zh9ug7inrpw593voagtit&st=v2hg4pt1&dl=0)
  - [Baidu Netdisk](https://pan.baidu.com/s/1FcCTWXbUy-O4ZfHN4n7PGQ?pwd=bcmi)
- **Description**: 
  - This dataset is built upon the existing [Foreground Object Search dataset](https://github.com/bcmi/Foreground-Object-Search-Dataset-FOSD).
  - Each background image within this dataset comes with a manually annotated bounding box. These bounding boxes are suitable for placing one object from a specified category.
  - The resultant dataset consists of 640 pairs of backgrounds and foregrounds. This dataset is utilized in our user study and qualitative comparison.

**We have extended FOSCom dataset to [MureCom](https://github.com/bcmi/DreamCom-Image-Composition/tree/main?tab=readme-ov-file#our-murecom-dataset) which has multiple images for one foreground object.**

## Code and Model

### 1.  Dependencies

  - Python == 3.8.5
  - Pytorch == 1.10.1
  - Pytorch-lightning == 1.9.0
  - Run

    ```bash
    cd ControlCom-Image-Composition
    pip install -r requirements.txt
    cd src/taming-transformers
    pip install -e .
    ```
### 2.  Download Models

  - Please download the following files to the ``checkpoints`` folder to create the following file tree:
    ```bash
    checkpoints/
    ├── ControlCom_blend_harm.pth
    ├── ControlCom_view_comp.pth
    └── openai-clip-vit-large-patch14
        ├── config.json
        ├── merges.txt
        ├── preprocessor_config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.json
    ```
  - **openai-clip-vit-large-patch14 ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/openai-clip-vit-large-patch14.zip) | [ModelScope](https://www.modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/openai-clip-vit-large-patch14.zip))**: The foreground encoder of our ControlCom is built on pretrained clip.

  - **ControlCom_blend_harm.pth ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/ControlCom_blend_harm.pth) | [ModelScope](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/ControlCom_blend_harm.pth))**: This model is finetuned for 20 epochs specifically for the tasks of image blending and harmonization. Therefore, when the ``task`` argument is set to "blending" or "harmonization" in the following test code, this checkpoint will be loaded.
  
  - **ControlCom_view_comp.pth ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/ControlCom_view_comp.pth) | [ModelScope](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/ControlCom_view_comp.pth))**: This model is enhanced on viewpoint transformation through finetuning for several epochs using additional multi-viewpoint datasets, *i.e.*, [MVImgNet](https://gaplab.cuhk.edu.cn/projects/MVImgNet/). When the ``task`` argument is set to "viewsynthesis" or "composition" in the following test code, this checkpoint will be loaded. Note tht this checkpoint can also be used for "blending" and "harmonization". If you wish to use one checkpoint for four tasks, we recommend this checkpoint. 


### 3. Inference on examples

  - To perform image composition using our model, you can use `scripts/inference.py`. For example,

    ```
    python scripts/inference.py \
    --task harmonization \
    --outdir results \
    --testdir examples \
    --num_samples 1 \
    --sample_steps 50 \
    --gpu 0
    ```
    or simply run:
    ```
    sh test.sh
    ```
These images under ``examples`` folder are obtained from [COCOEE](https://github.com/Fantasy-Studio/Paint-by-Example) dataset. 

### 4. Inference on your data

- Please refer to the [examples](./examples/) folder for data preparation:
  - keep the same filenames for each pair of data. 
  - either the ``mask_bbox`` folder or the ``bbox`` folder is sufficient. 
  - ``foreground_mask`` folder is optional but recommended for better composite results.

### 5. Training code

- **Download link**:
  - [Dropbox](https://www.dropbox.com/scl/fi/7xct03btipclhl8a8z135/ControlCom_train.zip?rlkey=hi1z5eh3b9kbtc4g4jrbjcq3q&st=nmuynh0z&dl=0)
  - [Baidu Netdisk](https://pan.baidu.com/s/1810soRtO9vxRpmJP51fNBA?pwd=zy1c)

**Notes**: certain sensitive information has been removed since the model training was conducted within a company. To start training, you'll need to prepare your own training data and make necessary modifications to the code according to your requirements.

## Experiments

We show our results using four types of indicators. 

<p align='center'>  
  <img src='./figures/controllable_results.jpg'  width=80% />
</p>

## Evaluation

The quantitative results and evaluation code can be found [here](https://github.com/bcmi/Awesome-Generative-Image-Composition?tab=readme-ov-file#leaderboard). 

## **Acknowledgements**
This code borrows heavily from [Paint-By-Example](https://github.com/Fantasy-Studio/Paint-by-Example). We also appreciate the contributions of [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

## Citation
If you find this work or code is helpful in your research, please cite:
````
@article{zhang2023controlcom,
  title={Controlcom: Controllable image composition using diffusion model},
  author={Zhang, Bo and Duan, Yuxuan and Lan, Jun and Hong, Yan and Zhu, Huijia and Wang, Weiqiang and Niu, Li},
  journal={arXiv preprint arXiv:2308.10040},
  year={2023}
}
````

## Other Resources
+ We summarize the papers and codes of generative image composition: [Awesome-Generative-Image-Composition](https://github.com/bcmi/Awesome-Generative-Image-Composition)
+ We summarize the papers and codes of image composition from all aspects: [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)
+ We summarize all possible evaluation metrics to evaluate the quality of composite images:  [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)
+ We write a comprehensive survey on image composition: [the latest version](https://arxiv.org/pdf/2106.14490.pdf)
