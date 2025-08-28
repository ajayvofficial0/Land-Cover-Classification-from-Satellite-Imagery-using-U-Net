# Land Cover Classification from Satellite Imagery using U-Net

**Authors:** V Ajay (ENG22AM0140), Sudin G Poojary (ENG22AM0135), Jeyadheep V (ENG22AM0141), Trijal R (ENG22AM0167)
**Supervisor:** Dr. Vinutha N, Associate Professor, CSE(AIML), DSU

---

## Overview

The rapid expansion of urbanization and agricultural activities has dramatically altered natural landscapes, making accurate and timely monitoring of land cover essential for environmental management, urban planning, and disaster mitigation. This project employs a U-Net convolutional neural network to perform pixel-wise semantic segmentation on high‑resolution satellite imagery, classifying each pixel into one of seven land cover categories (Urban, Agriculture, Rangeland, Forest, Water, Barren, and Unknown). By leveraging the DeepGlobe Land Cover Classification dataset, the model learns to discern subtle spatial patterns and textures, enabling automated land cover mapping at scale. Key contributions include robust data preprocessing pipelines, optional augmentation strategies to improve generalization, and comprehensive evaluation using metrics such as Pixel Accuracy, Intersection over Union (IoU), and Dice Coefficient. The resulting system not only demonstrates competitive quantitative performance but also provides qualitative visualizations that aid stakeholders in making informed decisions about land use and conservation.

## Table of Contents

* [Dataset](#dataset)
* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Methodology](#methodology)
* [Results](#results)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)

## Dataset

This project uses the **DeepGlobe Land Cover Classification Challenge** dataset, which comprises RGB satellite images (2448×2448 pixels) annotated with seven land cover classes. The dataset is publicly available via the CVPR 2018 challenge and requires preprocessing (resizing, normalization, encoding) before training.

## Features

* U-Net architecture with encoder–decoder and skip connections
* Data preprocessing: image resizing, normalization, one-hot encoding
* Optional data augmentation (flips, rotations, scaling)
* Training with categorical cross-entropy loss and Adam optimizer
* Evaluation using Pixel Accuracy, Intersection over Union (IoU), and Dice coefficient
* Visualization scripts for input images and predicted masks

## Requirements

* Python 3.7 or higher
* CUDA-compatible GPU (optional, but recommended)
* Key Python packages:

  * `torch`, `torchvision`
  * `numpy`, `opencv-python`
  * `matplotlib`, `pandas`, `scikit-learn`
  * `albumentations` (optional for augmentation)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/landcover-unet.git
   cd landcover-unet
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the DeepGlobe dataset and organize it under `data/`:

   ```text
   data/
   ├─ train/
   ├─ valid/
   └─ test/
   ```

## Usage

1. **Preprocess Dataset**:

   ```bash
   python scripts/preprocess.py --data_dir data/ --output_dir processed_data/ --img_size 256
   ```
2. **Train Model**:

   ```bash
   python train.py --data_dir processed_data/ --batch_size 16 --epochs 10 --lr 1e-3
   ```
3. **Evaluate Model**:

   ```bash
   python evaluate.py --model checkpoints/unet_epoch10.pth --data_dir processed_data/test/
   ```
4. **Visualize Predictions**:

   ```bash
   python visualize.py --model checkpoints/unet_epoch10.pth --samples 5 9
   ```

## Methodology

1. **Architecture**: Implemented U-Net with `DoubleConv` blocks, max pooling, and transposed convolutions for upsampling. Skip connections preserve spatial details.
2. **Training**: Used Adam optimizer (`lr=0.001`), `CrossEntropyLoss`, batch size 16, for 10 epochs.
3. **Evaluation**: Calculated Pixel Accuracy (\~76.09%), Mean IoU (\~22.42%), and Dice Coefficient (\~27.35%).

## Results

* Pixel Accuracy: **0.7609**
* Mean IoU: **0.2242**
* Dice Coefficient: **0.2735**

Qualitative examples of input images and predicted masks are available in the `visuals/` directory.

## Project Structure

* `DL_Report_Landcoverunet_Team-11.pdf` — Project report
* `Land Cover Unet_CODE.ipynb` — Implementation notebook
