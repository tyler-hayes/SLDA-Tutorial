Deep Streaming Linear Discriminant Analysis: A Tutorial
=====================================
This is a PyTorch tutorial on how to use the Deep Streaming Linear Discriminant Analysis (SLDA) algorithm from our CVPRW-2020 [paper](https://arxiv.org/abs/1909.01520).
This repo uses the original scenario definitions for the [CORe50 dataset](https://vlomonaco.github.io/core50/), which differ from the CORe50 experiments in our paper.
To replicate our ImageNet experiments, please see [this repo](https://github.com/tyler-hayes/Deep_SLDA).

## Dependences 
- Tested with Python 3.7 and PyTorch 1.3.1, NumPy, NVIDIA GPU
- Install the [Avalanche framework](https://github.com/ContinualAI/avalanche)
  
## Usage
To experiment with SLDA on CORe50, let's run:
- `main_tutorial.py`

## Citation
If using this code, please cite our paper.
```
@InProceedings{Hayes_2020_CVPR_Workshops,
    author = {Hayes, Tyler L. and Kanan, Christopher},
    title = {Lifelong Machine Learning With Deep Streaming Linear Discriminant Analysis},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2020}
}