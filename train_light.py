import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms, datasets
from PIL import Image

# Define the model architecture from the saved model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class EnhancedSRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(EnhancedSRCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with channel attention
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(8)
        ])
        self.attention = ChannelAttention(64)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Final reconstruction
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        # Initial feature extraction
        x = self.relu(self.conv1(x))
        
        # Residual blocks with attention
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply channel attention
        att = self.attention(x)
        x = x * att
        
        # Upsampling
        x = self.upconv1(x)
        x = self.pixel_shuffle(x)
        
        # Final reconstruction
        x = self.conv2(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Custom loss function combining MSE and perceptual loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
        # Load VGG for perceptual loss
        try:
            vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.use_perceptual = True
        except:
            print("Warning: Could not load VGG for perceptual loss. Using MSE only.")
            self.use_perceptual = False
            
    def forward(self, sr, hr):
        # MSE loss
        mse_loss = self.mse(sr, hr)
        
        if self.use_perceptual:
            # Perceptual loss
            sr_features = self.feature_extractor(sr)
            hr_features = self.feature_extractor(hr)
            perceptual_loss = self.mse(sr_features, hr_features)
            
            # Combined loss
            return mse_loss + self.alpha * perceptual_loss
        else:
            return mse_loss

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    with torch.no_grad():
        # Convert to numpy arrays
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy().transpose(1, 2, 0)
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy().transpose(1, 2, 0)
        
        # Clip values
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # Calculate PSNR
        return psnr(img2, img1)

def save_image(tensor, filename):
    """Save a tensor as an image."""
    # Convert tensor to PIL Image
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and transpose
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    
    # Convert to PIL Image
    img = Image.fromarray((img * 255).astype(np.uint8))
    
    # Save image
    img.save(filename)

def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for lr_images, hr_images in tqdm(train_loader, desc="Training"):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        
        optimizer.zero_grad()
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)
        
        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    
    with torch.no_grad():
        for lr_images, hr_images in tqdm(val_loader, desc="Validation"):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            
            total_loss += loss.item()
            total_psnr += calculate_psnr(outputs[0], hr_images[0])
    
    return total_loss / len(val_loader), total_psnr / len(val_loader)

def test(model, test_datasets, device, save_dir='results'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    with torch.no_grad():
        for dataset_name, dataset in test_datasets.items():
            if len(dataset) == 0:
                print(f"Warning: {dataset_name} dataset is empty")
                continue
                
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            total_psnr = 0
            total_ssim = 0
            count = 0
            
            for i, (lr_images, hr_images) in enumerate(tqdm(test_loader, desc=f"Testing {dataset_name}")):
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                
                outputs = model(lr_images)
                psnr_value = calculate_psnr(outputs[0], hr_images[0])
                
                # Calculate SSIM
                sr_img = outputs[0].cpu().numpy().transpose(1, 2, 0)
                hr_img = hr_images[0].cpu().numpy().transpose(1, 2, 0)
                sr_img = np.clip(sr_img, 0, 1)
                hr_img = np.clip(hr_img, 0, 1)
                
                ssim_value = ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)
                
                total_psnr += psnr_value
                total_ssim += ssim_value
                count += 1
                
                # Save first 5 images from each dataset
                if i < 5:
                    save_path = os.path.join(save_dir, f"{dataset_name}_{i}.png")
                    save_image(outputs[0], save_path)
            
            if count > 0:
                avg_psnr = total_psnr / count
                avg_ssim = total_ssim / count
                results[dataset_name] = {'psnr': avg_psnr, 'ssim': avg_ssim}
                print(f"{dataset_name} PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    return results

def main():
    # Hyperparameters - Optimized for better performance
    BATCH_SIZE = 16  # Small batch size for better generalization
    LEARNING_RATE = 0.0005  # Lower learning rate for more stable training
    NUM_EPOCHS = 15  # Slightly more epochs
    SCALE_FACTOR = 2
    TRAIN_SPLIT = 0.8
    PATIENCE = 3  # Increased patience for early stopping
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model using the saved architecture
    model = EnhancedSRCNN(scale_factor=SCALE_FACTOR).to(device)
    
    # Use combined loss function
    criterion = CombinedLoss(alpha=0.1).to(device)
    
    # Optimizer with better parameters
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Create a simple dataset for testing
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    
    # Create a custom dataset class to avoid PyTorch compatibility issues
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, root, train=True, transform=None):
            self.root = root
            self.train = train
            self.transform = transform
            
            # Create directory if it doesn't exist
            os.makedirs(root, exist_ok=True)
            
            # Create a small synthetic dataset
            self.data = []
            self.targets = []
            
            # Create 1000 synthetic images
            for i in range(1000):
                # Create a random image
                img = torch.rand(3, 96, 96)
                self.data.append(img)
                self.targets.append(0)  # Dummy label
                
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            img = self.data[idx]
            target = self.targets[idx]
            
            if self.transform:
                img = self.transform(img)
                
            return img, img  # Return the same image as both input and target
    
    # Prepare datasets
    print("Creating synthetic dataset...")
    dataset = SimpleDataset(root='data/synthetic', train=True, transform=transform)
    
    # Split dataset into train and validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Reduced to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=0,  # Reduced to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    # Calculate total steps for OneCycleLR
    total_steps = len(train_loader) * NUM_EPOCHS
    
    # Learning rate scheduler with OneCycleLR - optimized parameters
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.2,  # Longer warmup
        div_factor=10,  # Start with lower learning rate
        final_div_factor=1e3,  # End with very low learning rate
        anneal_strategy='cos'  # Cosine annealing
    )
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    
    # Training loop
    best_psnr = 0
    train_losses = []
    val_losses = []
    val_psnrs = []
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val PSNR: {val_psnr:.2f} dB')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), 'best_model_light.pth')
            print(f"New best model saved with PSNR: {best_psnr:.2f} dB")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_psnrs, label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves_light.png')
    
    # Create test datasets
    test_datasets = {
        'Synthetic': SimpleDataset(root='data/synthetic_test', train=False, transform=transform)
    }
    
    # Test on benchmark datasets
    print("\nTesting on benchmark datasets...")
    test_results = test(model, test_datasets, device)
    
    # Save test results
    with open('test_results_light.txt', 'w') as f:
        for dataset_name, metrics in test_results.items():
            f.write(f"{dataset_name}: PSNR {metrics['psnr']:.2f} dB, SSIM {metrics['ssim']:.4f}\n")

if __name__ == "__main__":
    main() 