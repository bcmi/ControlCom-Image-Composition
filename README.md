# ControlCom-Image-Composition

This is the official repository for the following research paper:

> **ControlCom: Controllable Image Composition using Diffusion Model**  [[arXiv]](https://arxiv.org/pdf/2308.10040.pdf)<br>
>
> Bo Zhang, Yuxuan Duan, Jun Lan, Yan Hong, Huijia Zhu, Weiqiang Wang, Li Niu<br>
>

## Task


In our controllable image composition model, we unify four tasks in one model using an 2-dim binary indicator vector, in which the first (*resp.*, second) dimension represents whether adjusting the foreground illumination (*resp.*, pose) to be compatible with background.  1 means making adjustment and 0 means remaining the same. Therefore, (0,0) corresponds to image blending, (1,0) corresponds to image harmonization, (0,1) corresponds to view synthesis, (1,1) corresponds to generative composition. 

<p align='center'>  
  <img src='./figures/task.png'  width=60% />
</p>

Our method can selectively adjust partial foreground attributes. Previous methods may adjust the foreground color/pose unexpectedly and even unreasonably, even when the foreground illumination and pose are already compatible with the background. In the left part, the foreground pose is already compatible with background and previous methods make unnecessary adjustment. In the right part, the foreground illumination is already compatible with the background and previous methods adjust the foreground color in an undesirable manner. 

<p align='center'>  
  <img src='./figures/controllability_necessity.jpg'  width=90% />
</p>

## Network Architecture

Our method is built upon stable diffusion and the network architecture is shown as follows.

<p align='center'>  
  <img src='./figures/architecture.png'  width=90% />
</p>

## Code and model

### 1.  Dependencies

  - Python == 3.8.5
  - Pytorch == 1.10.1
  - Pytorch-lightning == 1.9.0
  - Run

    ```bash
    cd ControlCom-Image-Composition
    pip install -r requirements.txt
    cd src/taming-transformers
    python setup.py install
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
  
  - **ControlCom_view_comp.pth ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/ControlCom_view_comp.pth) | [ModelScope](https://modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/ControlCom_view_comp.pth))**: This model is enhanced on viewpoint transformation through finetuning for several epochs using additional multi-viewpoint datasets, *i.e.*, [MVImgNet](https://gaplab.cuhk.edu.cn/projects/MVImgNet/). When the ``task`` argument is set to "viewsynthesis" or "composition" in the following test code, this checkpoint will be loaded.


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

### 4. Inference on your own data

- Please refer to the [examples](./examples/) folder for data preparation:
  - keep the same filenames for each pair of data. 
  - either the ``mask_bbox`` folder or the ``bbox`` folder is sufficient. 
  - ``foreground_mask`` folder is optional but recommended for better composite results.

**Part of our ControlCom has been integrated into our image composition toolbox libcom https://github.com/bcmi/libcom. Welcome to visit and try!** 

## Experiments

We show our results using four types of indicators. 

<p align='center'>  
  <img src='./figures/controllable_results.jpg'  width=60% />
</p>

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

+ We summarize the papers and codes of image composition from all aspects: [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)
+ We summarize all possible evaluation metrics to evaluate the quality of composite images:  [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)
+ We write a comprehensive on image composition: [the 3rd edition](https://arxiv.org/abs/2106.14490)
