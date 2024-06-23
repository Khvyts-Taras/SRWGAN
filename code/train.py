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


BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
SMALL_SIZE = 32
LARGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize(LARGE_SIZE),
    transforms.CenterCrop(LARGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



os.makedirs("samples", exist_ok=True)
os.makedirs("inputs", exist_ok=True)
os.makedirs("originals", exist_ok=True)
os.makedirs("models_big", exist_ok=True)

os.makedirs("/data/Places365", exist_ok=True)
dataset = datasets.Places365("/data/Places365", split='val', small=True, download=False, transform=transform)
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


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
        
        # Residual connection
        return output + F.interpolate(x, size=(128, 128), mode='bicubic')


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def create_small_images(images):
    return F.interpolate(images, size=SMALL_SIZE, mode='bicubic', align_corners=False)


def gradient_penalty(real, fake):
    m = real.shape[0]
    epsilon = torch.rand(m, 1, 1, 1)

    epsilon = epsilon.to(device)
    
    interpolated_img = epsilon * real + (1-epsilon) * fake
    interpolated_out = discriminator(interpolated_img)

    grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
                               grad_outputs=torch.ones(interpolated_out.shape).to(device),
                               create_graph=True, retain_graph=True)[0]
    grads = grads.reshape([m, -1])
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT) #pretrained=True
fe_net1 = vgg16.features[0:4].to(device)
# Инициализация моделей
generator = ImageUpscaler().to(device)
discriminator = Discriminator().to(device)

# Оптимизаторы
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

criterion = nn.MSELoss()


N_CRITIC = 5
GRADIENT_PENALTY = 10

g_losses = []
d_losses = []

for epoch in range(NUM_EPOCHS):
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    num_batches = len(data_loader)
    
    with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch') as pbar:
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            small_images = create_small_images(images)
            interpolated_small = F.interpolate(small_images, size=LARGE_SIZE, mode='bicubic')

            generated_images = generator(small_images)

            fake_output = discriminator(generated_images)
            real_output = discriminator(images)

            d_loss = (real_output.mean() - fake_output.mean()) + gradient_penalty(images, generated_images) * GRADIENT_PENALTY

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            d_loss_epoch += d_loss.item()

            if i % N_CRITIC == 0:
                generated_images = generator(small_images)
                fake_output = discriminator(generated_images)
                    
                g_loss = fake_output.mean()*0.01 + criterion(fe_net1(generated_images), fe_net1(images))

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
                g_loss_epoch += g_loss.item()
            


            avg_g_loss = g_loss_epoch / ((i // N_CRITIC) + 1)
            avg_d_loss = d_loss_epoch / (i + 1)
            pbar.set_postfix(g_loss=avg_g_loss, d_loss=avg_d_loss)
            pbar.update(1)

            if i%100 == 0:
                vutils.save_image(generated_images/2+0.5, os.path.join('samples', f'sample_{epoch}_{i}.png'))
                vutils.save_image(interpolated_small/2+0.5, os.path.join('inputs', f'sample_{epoch}_{i}.png'))
                vutils.save_image(images/2+0.5, os.path.join('originals', f'sample_{epoch}_{i}.png'))
    

    g_losses.append(g_loss_epoch / (num_batches // N_CRITIC))
    d_losses.append(d_loss_epoch / num_batches)


    print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{i}/{len(data_loader)}], "
          f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), f'models_big/generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'models_big/discriminator_epoch_{epoch}.pth')

print("Обучение завершено!")