import torch
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader
from models.deblurgan import Generator, Discriminator  # Ensure these models are defined in models/deblurgan.py
from datasets import ImageDataset  # Ensure this is properly defined in datasets.py

# Hyperparameters
epochs = 100
batch_size = 4
learning_rate = 0.0002
image_size = 256
save_path = 'checkpoints/'

# Create transformations for input images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prepare dataset and dataloader with correct file path (use raw strings or forward slashes)
dataset = ImageDataset(root_dir=r'C:\Users\user\Documents\Image processing-pytorch\models\Training', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

# Set up optimizers and loss functions
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
criterion_GAN = torch.nn.BCELoss()
criterion_L1 = torch.nn.L1Loss()

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        blurry_image, sharp_image = data['blurry'].to(device), data['sharp'].to(device)

        # Forward pass: Discriminator
        optimizer_D.zero_grad()

        # Pass the sharp image through the discriminator
        real_output = discriminator(sharp_image)

        # Ensure that the target size for real images matches the discriminator output
        real_target = torch.ones_like(real_output)
        real_loss = criterion_GAN(real_output, real_target)

        # Generate a fake image using the generator and pass it through the discriminator
        fake_image = generator(blurry_image)
        fake_output = discriminator(fake_image.detach())

        # Ensure that the target size for fake images matches the discriminator output
        fake_target = torch.zeros_like(fake_output)
        fake_loss = criterion_GAN(fake_output, fake_target)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Forward pass: Generator
        optimizer_G.zero_grad()

        # Generator loss based on GAN and L1 (content) loss
        g_loss_gan = criterion_GAN(discriminator(fake_image), torch.ones_like(fake_output))  # Target is real (1s)
        g_loss_l1 = criterion_L1(fake_image, sharp_image)  # Content loss
        g_loss = g_loss_gan + g_loss_l1 * 100
        g_loss.backward()
        optimizer_G.step()

        print(f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} Loss D: {d_loss.item()}, Loss G: {g_loss.item()}')

    # Save model checkpoints every 10 epochs
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), save_path + f'latest_net_G_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), save_path + f'latest_net_D_epoch_{epoch}.pth')

print("Training complete!")
