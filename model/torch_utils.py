from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path
from typing import Optional, Callable
from PIL import Image
import torch.nn as nn
from glob import glob
class AnimeDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None):
        print(f"[DEBUG] Initializing AnimeDataset ({split}) from {root_dir}", flush=True)
        self.root_dir = Path(root_dir)
        imgs = sorted(list(self.root_dir.glob("*.png")))
        print(f"[DEBUG] Found {len(imgs)} .png images", flush=True)

        if split == "train":
            self.image_paths = imgs[:int(len(imgs) * 0.99)]
        else:
            self.image_paths = imgs[int(len(imgs) * 0.99):]

        print(f"[DEBUG] Using {len(self.image_paths)} images for split={split}", flush=True)
        self.transform = transform


    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        
        return image, 0.0
    
    

class PadToSquare():
    def __call__(self,image: Image.Image) -> Image.Image:
        w, h  = image.size
        max_wh = max(w, h)
        pad_w = (max_wh - w) // 2
        pad_h = (max_wh - h) // 2
        padding = (pad_w, pad_h, max_wh - w - pad_w, max_wh - h - pad_h)
        return transforms.functional.pad(image,padding,fill=0, padding_mode='constant')
    
def load_data(args):
    
    print(f"[DEBUG] Total .png files found: {len(glob(args.dataset_path + '/*.png'))}", flush=True)

    train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            PadToSquare(), 
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    val_transforms = transforms.Compose([
            PadToSquare(), 
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    train_dataset = AnimeDataset(
            args.dataset_path,
            split='train',
            transform=train_transforms
        )
    val_dataset = AnimeDataset(
            args.dataset_path,
            split='val',
            transform=val_transforms
        )
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    return train_loader, val_loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)