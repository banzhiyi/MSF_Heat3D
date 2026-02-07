# MSF_Heat3D

本仓库提供 **MSF_Heat3D** 的训练与测试代码，以及在高光谱数据集上的实验复现流程。

---

## 1. 环境配置（Environment Setup）

### 1.1 Python 版本
- Python == **3.10**

建议使用 Conda 创建独立环境：
```bash
conda create -n msfheat3d python=3.10 -y
conda activate msfheat3d
```

### 1.2 安装 PyTorch（CUDA 12.1）
请按如下命令安装指定版本（cu121）：
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 安装 timm
```bash
pip install "timm>=0.9.12"
```

### 1.4 配置 mamba_ssm 与 causal_conv1d
本项目依赖 **mamba_ssm** 与 **causal_conv1d**。请根据你的 CUDA / 编译环境进行安装与编译配置。

> 提示：这两个库通常包含 CUDA/CPP 扩展，若安装报错，请确认：
> - 已正确安装 CUDA（与当前 PyTorch 的 CUDA 版本匹配）
> - 本机具备可用的 C++ 编译器与 NVCC
> - 已升级 pip/setuptools/wheel

---

## 2. 数据集准备（Dataset Preparation）

1. 下载所需数据集（例如：Indian Pines）。
2. 在项目根目录下创建 `data` 文件夹。
3. 将下载好的数据集文件放入 `data` 目录下。

示例目录结构：
```text
data/
  IndianPine.mat
```

---

## 3. 运行方式（Usage）

### 3.1 训练（Train）
```bash
python demo.py --dataset='Indian' --patches=7 --flag_test='train' --model_name='MSF_Heat3D' --epoches=500 --batch_size=64 --train_ratio=1.0
```

### 3.2 测试（Test）
```bash
python demo.py --dataset='Indian' --patches=7 --flag_test='test' --model_name='MSF_Heat3D' --epoches=500 --batch_size=64
```

---

## 4. 说明（Notes）
- 请确保数据集已正确放置在 `data/` 目录下，否则训练/测试将无法读取数据。
- 若在安装 `mamba_ssm` / `causal_conv1d` 时遇到编译问题，请优先检查 CUDA、编译器以及 PyTorch CUDA 版本是否匹配。
