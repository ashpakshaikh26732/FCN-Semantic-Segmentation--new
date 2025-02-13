# Fully Convolutional Networks for Semantic Segmentation

## Overview

This project implements a Fully Convolutional Network (FCN) for semantic segmentation. FCNs are designed to classify each pixel in an image, making them suitable for applications like medical imaging, autonomous driving, and scene understanding.

## Dataset

The model is trained on a dataset of segmented images. Ensure that your dataset is organized in the following structure:

```
/data
    /images  (input images)
    /masks   (corresponding segmentation masks)
```
## 📹 Video Demonstrations

You can check out the original and inference videos here:

🔗 **Original Video:** [Click to View](https://drive.google.com/file/d/18C_C5O0nEausyeeg9zPOhDAFIFSFddBa/view?usp=drive_link)  
🔗 **Inference Video:** [Click to View](https://drive.google.com/file/d/18C_C5O0nEausyeeg9zPOhDAFIFSFddBa/view?usp=drive_link)  

You can use datasets like Pascal VOC, Cityscapes, or a custom dataset.

## Model Architecture

The FCN model is based on an encoder-decoder structure:

- **Encoder:** Pretrained CNN (e.g., VGG16, ResNet) extracts feature maps.
- **Decoder:** Upsampling layers restore spatial resolution for pixel-wise classification.

## Installation

Clone the repository and install dependencies:

```sh
!git clone https://github.com/ashpakshaikh26732/Fully-Convolutional-Neural-Networks-for-Image-Segmentation.git
cd Fully-Convolutional-Neural-Networks-for-Image-Segmentation
pip install -r requirements.txt
```

## Usage

### Training the Model

```sh
python train.py --epochs 50 --batch_size 16 --dataset_path ./data
```

### Testing the Model

```sh
python test.py --model_path ./checkpoints/model.pth --test_data ./data/test
```

### Visualizing Results

```sh
python visualize.py --model_path ./checkpoints/model.pth --sample_image ./data/sample.jpg
```

## Results

Example output:
![Sample Segmentation Output](./results/sample_output.png)

## Future Improvements

- Experimenting with different backbone networks
- Optimizing training with better data augmentation
- Deploying the model for real-time segmentation

## References

- Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015)
- TensorFlow & PyTorch implementations of FCNs
