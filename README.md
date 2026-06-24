# VPSentry
Official PyTorch implementation of "VPSentry: Semi-supervised Video Polyp Segmentation via Sentry-guided Long-term Prototype Fusion with Correlation Dynamic Propagation".

# Getting Start
## Installation
**1. Envs: python=3.10.13 and CUDA=11.8**
```
conda create -n vpsentry python=3.10.13
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install opencv
pip install -U scikit-learn
cd ..work_path../VPSentry
```
**2. Requirements**
```
pip install -r ..work_path../requirements.txt
```
**3.Traning**
