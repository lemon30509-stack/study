# 基于GAN的少样本胸部X光肺炎识别模型

本仓库基于系统设计文档实现了可运行的项目骨架，覆盖：

- 数据预处理与少样本采样
- WGAN-GP 生成模型训练
- ResNet-18 肺炎分类模型微调
- 评估与推理
- Gradio 可视化界面

## 怎么运行（最短路径）

### 0) 安装环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) 准备数据目录

原始数据目录需要是 `ImageFolder` 结构（Kaggle chest_xray 常见结构）：

```text
data/raw/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

### 2) 预处理 + 少样本采样

```bash
python data_preprocess.py \
  --input_dir data/raw/chest_xray \
  --output_dir data/processed \
  --per_class 100
```

### 3) 训练 WGAN-GP

```bash
python train_gan.py \
  --data_dir data/processed/train \
  --epochs 200 \
  --output_dir outputs/gan
```

> 说明：当前 `train_gan.py` 会在 `outputs/gan/samples` 下保存网格图。如果你希望把 GAN 图像用于分类训练，请把筛选后的合成图整理为如下结构：

```text
outputs/gan_for_cls/
  NORMAL/
  PNEUMONIA/
```

### 4) 训练分类器（支持可选 GAN 合成数据）

不用 GAN 合成数据：

```bash
python train_cls.py \
  --train_dir data/processed/train \
  --val_dir data/processed/val \
  --test_dir data/processed/test
```

使用 GAN 合成数据（可选）：

```bash
python train_cls.py \
  --train_dir data/processed/train \
  --val_dir data/processed/val \
  --test_dir data/processed/test \
  --synthetic_dir outputs/gan_for_cls
```

### 5) 评估与推理

```bash
python evaluate.py --ckpt outputs/classifier/best_classifier.pth --data_dir data/processed/test
python inference.py --ckpt outputs/classifier/best_classifier.pth --image demo.png
```

### 6) 启动可视化界面

```bash
python ui_app.py
```

## 目录结构

- `config.py`：默认超参数配置
- `utils.py`：通用工具函数（种子、指标、保存/加载）
- `data_preprocess.py`：预处理、划分、少样本采样
- `train_gan.py`：WGAN-GP 训练
- `train_cls.py`：ResNet-18 微调训练
- `evaluate.py`：测试集指标与混淆矩阵
- `inference.py`：单图推理
- `ui_app.py`：Gradio 页面
