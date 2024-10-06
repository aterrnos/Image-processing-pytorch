import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.blurry_images = os.listdir(os.path.join(root_dir, 'blurry'))
        self.sharpened_images = os.listdir(os.path.join(root_dir, 'sharpened'))

    def __len__(self):
        return len(self.blurry_images)

    def __getitem__(self, idx):
        blurry_image_path = os.path.join(self.root_dir, 'blurry', self.blurry_images[idx])
        sharpened_image_path = os.path.join(self.root_dir, 'sharpened', self.sharpened_images[idx])

        blurry_image = Image.open(blurry_image_path).convert("RGB")
        sharpened_image = Image.open(sharpened_image_path).convert("RGB")

        if self.transform:
            blurry_image = self.transform(blurry_image)
            sharpened_image = self.transform(sharpened_image)

        return {'blurry': blurry_image, 'sharp': sharpened_image}
