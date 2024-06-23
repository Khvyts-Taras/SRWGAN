import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils
from torch import autograd
from tqdm import tqdm
import torchvision.models as models
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ImageUpscaler(nn.Module):
    def __init__(self):
        super(ImageUpscaler, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 16)
        self.enc2 = self.conv_block(16, 32)
        
        # Bottleneck
        self.bottleneck = self.conv_block(32, 64)
        
        # Decoder (upsampling)
        self.dec2 = self.upconv_block(64, 32)
        self.dec1 = self.upconv_block(64, 16)  # 64 because of skip connection
        
        # Additional upsampling layers
        self.up1 = self.upconv_block(16, 16)
        self.up2 = self.upconv_block(16, 16)
        
        # Final convolution
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1, padding_mode='replicate')
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.SELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.SELU(),
            nn.InstanceNorm2d(out_channels)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.SELU(),
            nn.InstanceNorm2d(out_channels)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # 32x32 -> 32x32
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # 32x32 -> 16x16
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))  # 16x16 -> 8x8
        
        # Decoder with skip connections
        dec2 = self.dec2(bottleneck)  # 8x8 -> 16x16
        dec1 = self.dec1(torch.cat([dec2, enc2], dim=1))  # 16x16 -> 32x32
        
        # Additional upsampling
        up1 = self.up1(dec1)  # 32x32 -> 64x64
        up2 = self.up2(up1)  # 64x64 -> 128x128
        
        # Final convolution
        output = self.final_conv(up2)  # 128x128 -> 128x128
        output = torch.tanh(output)

        print(output.shape[2:])
        
        # Residual connection
        return output + F.interpolate(x, size=output.shape[2:], mode='bicubic')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = ImageUpscaler().to(device)

# Load model from /models_big/generator_epoch_22.pth
model_path = 'models_big/generator_epoch_22.pth'
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Function to process a single image
def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = generator(img_tensor)
    
    # Denormalize and convert to PIL Image
    output = output * 0.5 + 0.5
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = (output * 255).clip(0, 255).astype('uint8')
    output_img = Image.fromarray(output)
    
    return output_img

# Process all images in /test_images and save to /result
input_folder = 'test_images'
output_folder = 'result'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'upscaled_{filename}')
        
        upscaled_img = process_image(input_path)
        upscaled_img.save(output_path)
        print(f'Processed and saved: {output_path}')

print('All images have been processed and saved.')