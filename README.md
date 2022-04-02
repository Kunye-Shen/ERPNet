# ERPNet (TCYB 2022)
Edge-guided Recurrent Positioning Network for Salient Object Detection in Optical Remote Sensing Images, [Xiaofei Zhou](https://scholar.google.com.hk/citations?hl=zh-CN&user=2PUAFW8AAAAJ), [Kunye Shen](https://scholar.google.com.hk/citations?hl=zh-CN&user=q6_PkywAAAAJ), Li Weng, Runmin Cong, Bolun Zheng, Jiyong Zhang, and Chenggang Yan.

## Required libraries

Python 3.7  
numpy 1.18.1  
scikit-image 0.16.2  
PyTorch 1.4.0  
torchvision 0.5.0  
glob

## Usage
1. Clone this repo
```
https://github.com/Kunye-Shen/ERPNet.git
```
We also provide the predicted saliency maps ([GoogleDrive](https://drive.google.com/drive/folders/1KV0DOnu_u6GlRT8qRfYc5ZC46RR5RQbI?usp=sharing) or [baidu](https://pan.baidu.com/s/1l3idpr-RGFz6KIVqBHKsHQ) extraction code: 8z2p.)

## Architecture
![EMFINet architecture](figures/architecture.png)

## Quantitative Comparison
![Quantitative Comparison](figures/quan.png)

## Qualitative Comparison
### ORSSD
![ORSSD](figures/qual_ORSSD.png)

### EORSSD
![EORSSD](figures/qual_EORSSD.png)
