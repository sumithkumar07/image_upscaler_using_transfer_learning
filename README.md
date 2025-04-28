# Image Upscaler using Transfer Learning

This project implements an enhanced SRCNN (Super-Resolution Convolutional Neural Network) model for image upscaling using transfer learning techniques. The model is designed to upscale images by a factor of 2x while preserving details and improving overall image quality.

## Features

- Enhanced SRCNN architecture with residual blocks and channel attention
- Transfer learning approach for better performance
- Streamlit web interface for easy image upscaling
- Quality metrics calculation (PSNR and SSIM)
- Support for various image formats (JPG, JPEG, PNG)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sumithkumar07/image_upscaler_using_transfer_learning.git
cd image_upscaler_using_transfer_learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at http://localhost:8501. You can:
- Upload an image
- See the original and upscaled versions side by side
- View quality metrics
- Download the upscaled image

### Model Architecture

The EnhancedSRCNN model includes:
- Residual blocks for improved gradient flow
- Channel attention mechanism for focusing on important features
- Pixel shuffle upsampling for efficient resolution increase
- Combined loss function (MSE + perceptual loss)

## Project Structure

- `app.py`: Streamlit web interface
- `train_light.py`: Training script with enhanced architecture
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DIV2K dataset for training
- SRCNN paper for the base architecture
- PyTorch team for the deep learning framework 