# Aerial Road Segmentation using PyTorch

![Aerial_Image_Segmentation](https://github.com/user-attachments/assets/62ab7e9c-dbcb-4bf8-a201-43b471495dff)

## Overview
This project implements a road segmentation model using a U-Net architecture with an EfficientNet-B0 backbone to detect and segment roads from aerial imagery. The project uses the Massachusetts Roads Dataset subset and demonstrates a complete deep learning workflow for image segmentation.

## Dataset
- **Source**: Massachusetts Roads Dataset (subset)
- **Images**: 200 aerial images
- **Image Size**: 1500×1500 pixels
- **Coverage**: 2.25 square kilometers per image
- **Task**: Binary segmentation of road networks

## Dependencies
- PyTorch
- segmentation-models-pytorch
- albumentations
- OpenCV
- NumPy
- Pandas
- scikit-learn
- matplotlib

## Project Structure

### Key Components
1. **Data Preparation**
   - Custom dataset loader
   - Train/validation split
   - Data augmentation using Albumentations

2. **Model Architecture**
   - U-Net with EfficientNet-B0 backbone
   - Binary segmentation
   - Uses pre-trained ImageNet weights

3. **Training Pipeline**
   - Adam optimizer
   - Dice Loss + Binary Cross Entropy
   - 25 training epochs
   - Model checkpoint saving

## Hyperparameters
- **Learning Rate**: 0.003
- **Batch Size**: 8
- **Image Size**: 512×512
- **Encoder**: timm-efficientnet-b0
- **Encoder Weights**: ImageNet

## Usage

### Installation
```bash
pip install segmentation-models-pytorch
pip install -U git+https://github.com/albumentations-team/albumentations
pip install --upgrade opencv-contrib-python
```

### Training
1. Clone the repository
2. Prepare your dataset
3. Adjust hyperparameters in the script
4. Run the training script

### Inference
- Load the trained model
- Use the `helper.show_image()` function to visualize predictions

## Results
- Visualizes road segmentation masks
- Demonstrates image augmentation techniques
- Provides a complete deep learning workflow

## Future Work
- Train on full Massachusetts Roads Dataset
- Experiment with different encoders and architectures
- Add more advanced augmentation techniques

## References
- [Segmentation Models PyTorch](https://smp.readthedocs.io/en/latest/)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- Massachusetts Roads Dataset

## License
MIT License

Feel free to use this implementation and modify it according to your needs. Contributions are welcome!
