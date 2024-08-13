import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)  # List all file names in the directory

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64 (The 50k Celeba Dataset already has 64x64 Images)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def getDataLoader(dir: str= "./50k", batch_size: int=128):
    dataset = ImageDataset(img_dir=dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

