import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils
from torch import autograd

# Определение констант
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
SMALL_SIZE = 32
LARGE_SIZE = 128
device = 'cuda'

# Определение трансформаций
transform = transforms.Compose([
    transforms.Resize(LARGE_SIZE),
    transforms.CenterCrop(LARGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Загрузка данных
os.makedirs("/data/Places365", exist_ok=True)
dataset = datasets.Places365("/data/Places365", split='val', small=True, download=False, transform=transform)
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

class ImageUpscaler(nn.Module):
    def __init__(self):
        super(ImageUpscaler, self).__init__()
        
        # Начальный свёрточный слой
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # Блок с повышением разрешения
        self.upscale_block = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Финальный свёрточный слой
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.upscale_block(x)
        x = self.final_conv(x)
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Свёрточные слои
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Полносвязный слой для классификации
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Инициализация моделей
generator = ImageUpscaler().to(device)
discriminator = Discriminator().to(device)

# Оптимизаторы
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Функции потерь
criterion = nn.BCELoss()

# Функция для создания маленьких изображений
def create_small_images(images):
    return F.interpolate(images, size=SMALL_SIZE, mode='bicubic', align_corners=False)

# Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


N_CRITIC = 3
GRADIENT_PENALTY = 10
for epoch in range(NUM_EPOCHS):
    for i, (images, _) in enumerate(data_loader):


        images = images.to(device)
        small_images = create_small_images(images)
        interpolated_small = F.interpolate(small_images, size=LARGE_SIZE, mode='nearest')

        generated_images = generator(small_images)

        fake_input = torch.cat((interpolated_small, generated_images), dim=1)
        real_input = torch.cat((interpolated_small, images), dim=1)

        fake_output = discriminator(fake_input)
        real_output = discriminator(real_input)

        d_loss = (real_output.mean() - fake_output.mean()) + gradient_penalty(real_input, fake_input) * GRADIENT_PENALTY

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()


        if i%N_CRITIC or i == 0:
            generated_images = generator(small_images)
            fake_input = torch.cat((interpolated_small, generated_images), dim=1)
            fake_output = discriminator(fake_input)
                

            g_loss = fake_output.mean()

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()


        if i % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{i}/{len(data_loader)}], "
                  f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

            vutils.save_image(generated_images/2+0.5, os.path.join('samples', f'sample_{epoch}_{i}.png'))

    # Сохранение моделей
    torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')

print("Обучение завершено!")