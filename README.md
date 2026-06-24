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
```
**2. Requirements**
```
pip install -r ..work_path../requirements.txt
```
**3.Traning and Testing**
```
//Train:
cd ..work_path../VPSentry
python /..work_path../VPSentry/scripts/my_train.py
//Test:
python /..work_path../VPSentry/scripts/my_test.py
```

### Logs and Weights
We provide the relevant logs and ckpts (trained on SUN-SEG dataset) based on two different backbones:
With 20% label:
[Res2Net-50]() and [PVTv2-B2]()
With 10% label:
[Res2Net-50]() and [PVTv2-B2]()

# Acknowledgement
Our work builds upon the excellent foundational research of [PNS+](https://github.com/GewelsJI/VPS) and [DPA]([https://github.com/hustvl/Vim](https://github.com/Hydragon516/DPA)). We thank the authors for their awesome works and publicly available codes.

# Citation
if you find our work useful, please cite:
```
@inproceedings{chen2026vpsentry,
  title={VPSentry: Semi-supervised Video Polyp Segmentation via Sentry-guided Long-term Prototype Fusion with Correlation Dynamic Propagation},
  author={Chen, Guilian and Luo, Xiaoling and Wu, Huisi and Qin, Jing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={4},
  pages={2850--2858},
  year={2026}
}
```
