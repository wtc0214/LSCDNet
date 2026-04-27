# LSCDNet：一种用于肾脏肿瘤与肾病检测的轻量级实时检测框架

## 项目简介
LSCDNet 是基于 YOLOv8 改进的轻量级实时目标检测框架，主要面向肾脏肿瘤与肾病检测任务。
针对现有检测方法存在的计算复杂度高、推理延迟大等问题，LSCDNet 重点优化了检测精度与实时性能之间的平衡，适用于临床诊断系统及边缘设备部署场景

## 模型结构
LSCDNet 引入了三个核心模块以提升检测效率与实时性：
- 轻量动态卷积（LDConv）
  通过自适应采样机制增强对不规则病灶区域的建模能力，同时减少冗余计算。
- 轻量共享卷积检测头（LSCD）
  在多尺度特征之间共享参数，减少重复计算，显著降低推理延迟。
- 空间组增强模块（SGE）
  在几乎不增加计算量的前提下，提高特征表达能力。


## 数据集

本项目在多个数据集上进行实验验证：

1. Kidney Tumor Dataset

🔗 https://universe.roboflow.com/tezskb/kidney-tumor-uqpis

2. Kidney Disease Dataset

🔗 https://universe.roboflow.com/kidney-disease-7tmoy/kidney-disease-gd4is

3. Liver Disease Dataset (Generalization)

🔗 https://universe.roboflow.com/roboflow-100/liver-disease



### 3. Install Dependencies
(环境安装推荐直接使用已配置好的 YOLOv8 或 YOLOv11 环境，无需重复安装）
```bash
# Step 1.Create a virtual environment with conda
conda create -n pt121_py38 python=3.8
conda activate pt121_py38

# Step 2: Install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# Step 3: Install the remaining dependencies

pip install -r requirements.txt


# https://pytorch.org/get-started/previous-versions/
## CUDA 10.2
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
## CUDA 11.3
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
## CUDA 11.6
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
## CPU Only
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch

## CUDA 11.8
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
## CUDA 12.1
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
## CPU Only
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```


### 4. 运行训练
```bash
python train.py --data your_dataset_config.yaml
```
#### 训练脚本说明

本项目包含多个训练脚本，适用于不同任务：

4.1. **`train.py`**
  - 基础训练脚本，适用于通用目标检测任务


4.2. **`train-rtdetr.py`**
   - 用于 RT-DETR 模型的训练

4.3. **`train_Gray.py`**
   - 灰度图训练脚本，适用于单通道图像任务


### 5.测试与验证

运行以下命令进行模型验证：
```bash
python val.py
```
