# main.py

# Import everything from DnCNN.py
from DnCNN import DnCNN

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import the Generator model from the DeblurGAN code
from models.deblurgan import Generator

# Path to your pre-trained weights
model_path = 'C:\Users\user\Documents\Image processing-pytorch\model'

# Initialize the model
model = Generator()

# Load pre-trained weights
model.load_state_dict(torch.load(model_path, map_location='cuda'))

# Switch model to evaluation mode
model.eval()


# Path to your pre-trained weights (update this to the correct path)
model_path = 'models/deblurgan.pth'  # Make sure this points to the correct weight file in the models folder

# Initialize the model
model = Generator()

# Load pre-trained weights
model.load_state_dict(torch.load(model_path, map_location='cuda'))


# Initialize the model
model = DnCNN(channels=3)  # Assuming DnCNN has a channels argument

# Load pre-trained weights if available
try:
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained weights successfully.")
except FileNotFoundError:
    print(f"FileNotFoundError: No such file or directory: '{model_path}'")
    # Optionally, continue running with untrained model
    print("Running without pre-trained weights.")

# Switch model to evaluation mode
model.eval()

# Load and preprocess the image
input_image_path = r'C:\Users\user\Documents\Image processing-pytorch\blurry\blurry.jpg'
input_image = Image.open(input_image_path).convert('RGB')

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(input_image).unsqueeze(0)  # Add a batch dimension

# Denoise the image using the model
if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

with torch.no_grad():
    output_tensor = model(input_tensor)

# Reverse normalization for visualization
reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
])

# Move tensor back to CPU and reverse the normalization
output_image = reverse_transform(output_tensor.squeeze().cpu()).permute(1, 2, 0).numpy()

# Clip values to [0, 1]
output_image = np.clip(output_image, 0, 1)

# Convert to image format and save
output_image_pil = Image.fromarray((output_image * 255).astype(np.uint8))
output_image_pil.save(r'C:\Users\user\Pictures\blurry\dncnn_denoised_image.png')
print("Denoised image saved successfully.")
