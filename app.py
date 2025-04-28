import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from train_light import EnhancedSRCNN
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Image Upscaling App",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ Image Upscaling with EnhancedSRCNN")
st.markdown("""
This app uses an enhanced SRCNN model to upscale images by 2x while preserving details and improving quality.
Upload an image below to see the results!
""")

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedSRCNN(scale_factor=2).to(device)
    model.load_state_dict(torch.load('best_model_light.pth', map_location=device))
    model.eval()
    return model, device

# Image processing functions
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def postprocess_image(tensor):
    # Check if input is a PyTorch tensor or NumPy array
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        img = tensor.cpu().numpy().transpose(1, 2, 0)
    else:  # It's already a NumPy array
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        img = tensor.transpose(1, 2, 0)
    
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

def calculate_metrics(original, upscaled):
    # Resize original to match upscaled size
    original_resized = original.resize((upscaled.shape[1], upscaled.shape[0]))
    original_np = np.array(original_resized) / 255.0
    upscaled_np = upscaled / 255.0
    
    # Calculate metrics
    psnr_value = psnr(original_np, upscaled_np, data_range=1.0)
    ssim_value = ssim(original_np, upscaled_np, channel_axis=2, data_range=1.0)
    
    return psnr_value, ssim_value

# Load model
model, device = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display original image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # Create two columns for original and upscaled images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
        st.write(f"Size: {original_image.size}")
    
    # Process image
    with torch.no_grad():
        # Preprocess
        input_tensor = preprocess_image(original_image).to(device)
        
        # Upscale
        output_tensor = model(input_tensor)
        
        # Postprocess
        upscaled_image = postprocess_image(output_tensor[0].cpu().numpy())
        
        # Calculate metrics
        psnr_value, ssim_value = calculate_metrics(original_image, np.array(upscaled_image))
        
        # Display results
        with col2:
            st.subheader("Upscaled Image (2x)")
            st.image(upscaled_image, use_container_width=True)
            st.write(f"Size: {upscaled_image.size}")
            
            # Display metrics
            st.markdown("### Quality Metrics")
            st.write(f"PSNR: {psnr_value:.2f} dB")
            st.write(f"SSIM: {ssim_value:.4f}")
            
            # Download button
            buf = BytesIO()
            upscaled_image.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Upscaled Image",
                data=byte_im,
                file_name="upscaled_image.png",
                mime="image/png"
            )

# Add footer
st.markdown("---")
st.markdown("""
### About the Model
This app uses an EnhancedSRCNN model with the following features:
- Residual blocks for improved gradient flow
- Channel attention mechanism
- Pixel shuffle upsampling
- Combined loss function (MSE + perceptual loss)
""") 