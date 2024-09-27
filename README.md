
# Medical Image Segmentation with U-Net and Custom Modules
This repository contains an implementation of the U-Net architecture for medical image segmentation, enhanced with custom modules from a research paper. The code is designed to work with the BUSI (Breast Ultrasound Images Dataset), which consists of ultrasound images of breast cancer

## Dataset

The BUSI dataset is available on Kaggle. It consists of three classes: normal, benign, and malignant. The dataset includes images along with their corresponding masks

## Links for Dataset
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

## Directory Structure

The dataset images are stored in the following structure:

```bash
data/
│
├── imgs/            # Ultrasound images
└── masks/           # Corresponding masks

```
All images are preprocessed to ensure they have equal dimensions using the script masks.py.

## Custom Modules From Research Paper

#### R_MLP (r_mlp.py) :-

The R_MLP module implements a rolling MLP-based operation that processes the image spatially along one dimension (width or height). It applies a learned linear projection at each spatial location after rolling the input

#### OR_MLP and DOR_MLP (or_dor.py) :-

The OR_MLP module performs orthogonal rolling along two dimensions: width and height. The output from both operations is combined to capture features across both axes. The DOR_MLP (Dual OR_MLP) combines two OR_MLP layers to further enhance the feature extraction capability along both dimensions

#### Lo2_Block (Lo2_Block.py) :-

The Lo2_Block integrates depthwise separable convolutions with DOR_MLP to achieve both local and global feature extraction. The Lo2_Block is added to the U-Net architecture to improve segmentation accuracy

#### Mask Processing (masks.py) :-
This script ensures that all images and their corresponding masks are resized to the same dimensions, which is crucial for consistency during training and evaluation

#### Changes to train.py :-
The training script (train.py) has been modified to:

 1. Add functionality for handling custom layers like R_MLP, OR_MLP, and Lo2_Block.

 2. Integrate the BUSI dataset for breast ultrasound image segmentation.

 3. Include performance metrics like Accuracy, F1-Score, and IoU (Intersection over Union)to evaluate the segmentation performance.


## How to Run the Model

1. Clone the Repository


```bash
git clone https://github.com/your-username/medical_segmentation.git
cd medical_segmentation

```
2. Dataset Setup

Download the BUSI dataset from Kaggle and place the images in the data/imgs folder, and the masks in the data/masks folder.

3. Train the Model

To start training, run the following command:

```bash
python train.py --epochs 10 --batch-size 8 --learning-rate 0.0001 --validation 10.0 --scale 0.5 --classes 2

```

4. Checkpoints and Logging

The model checkpoints and logs are automatically saved, and the training process can be monitored using WandB (Weights and Biases). Ensure you log in to WandB before running the script.


## Output Metrics

After training, the model outputs several performance metrics:

 1. Accuracy: Measures the percentage of correctly predicted pixels.

 2. F1-Score: The harmonic mean of precision and recall, indicating the balance between them.

 3. IoU (Intersection over Union): Measures the overlap between predicted and ground truth segmentation masks.

 4. Loss: The training and validation loss is tracked over epochs.