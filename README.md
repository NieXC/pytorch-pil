
# Human Pose Estimation with Parsing Induced Learner

This repository contains the code and pretrained models of
> **Human Pose Estimation with Parsing Induced Learner** [[PDF](http://openaccess.thecvf.com/content_cvpr_2018/papers/Nie_Human_Pose_Estimation_CVPR_2018_paper.pdf)]     
> **Xuecheng Nie**, Jiashi Feng, Yiming Zuo, and Shuicheng Yan   
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018    

## Prerequisites

- Python 3.5
- Pytorch 0.2.0
- OpenCV 3.0 or higher

## Installation

1. Install Pytorch: Please follow the [official instruction](https://pytorch.org/) on installation of Pytorch.
2. Clone the repository   
   ```
   git clone --recursive https://github.com/NieXC/pytorch-pil.git
   ``` 
3. Download [Look into Person (LIP)](http://sysu-hcp.net/lip/overview.php) dataset and create symbolic links to the following directories
   ```
   ln -s PATH_TO_LIP_TRAIN_IMAGES_DIR dataset/lip/train_images   
   ln -s PATH_TO_LIP_VAL_IMAGES_DIR dataset/lip/val_images      
   ln -s PATH_TO_LIP_TEST_IMAGES_DIR dataset/lip/testing_images   
   ln -s PATH_TO_LIP_TRAIN_SEGMENTATION_ANNO_DIR dataset/lip/train_segmentations   
   ln -s PATH_TO_LIP_VAL_SEGMENTATION_ANNO_DIR dataset/lip/val_segmentations   
   ```

## Citation

If you use our code/model in your work or find it is helpful, please cite the paper:
```
@inproceedings{nie2018pil,
  title={Human Pose Estimation with Parsing Induced Learner},
  author={Nie, Xuecheng and Feng, Jiashi and Zuo, Yiming and Yan, Shuicheng},
  booktitle={CVPR},
  year={2018}
}
```
