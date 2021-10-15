import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch
import os
from robustness.tools import folder
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, ds, primary_transform, augmentation_transform=None):
        self.ds = ds
        self.tensor_to_image = transforms.ToPILImage()
        self.primary_transform = primary_transform
        self.augmentation_transform = augmentation_transform
        self.image_to_tensor = transforms.ToTensor()
        self.transform_cache = {}
        self.populate_cache()

    def __len__(self):
        return len(self.ds)

    def get_val(self, idx):
        vals = self.ds[idx]
        if len(vals) == 3:
            return vals
        else:
            x, y = vals
            return x, y, {}

    def populate_cache(self):
        if self.augmentation_transform is not None:
            print("populating cache")
            for idx in tqdm(range(len(self))):
                x_orig, _, _ = self.get_val(idx)
                x_orig = self.tensor_to_image(x_orig)
                x = self.augmentation_transform(x_orig)
                self.transform_cache[idx] = x

    def __getitem__(self, idx):
        x_orig, y, meta = self.get_val(idx)
        x_orig = self.tensor_to_image(x_orig)
        
        if self.augmentation_transform is not None:
            x = self.transform_cache[idx]
        else:
            x = x_orig
  
        x = self.primary_transform(x)

        x_orig = self.image_to_tensor(x_orig)
        x = self.image_to_tensor(x)

        if isinstance(y, int):
            y = torch.tensor(y)
        meta['original_img'] = x_orig
        return x, y, meta

def make_wrappers(datasets, primary_transform, additional_transform):
    all_ds = []
    for ds in datasets:
        all_ds.append(DatasetWrapper(ds, primary_transform=primary_transform,
                                     augmentation_transform=additional_transform))
    ds = torch.utils.data.dataset.ConcatDataset(all_ds)
    return ds

def get_loader(dataset, model_args, shuffle=True, drop_last=True):
    return DataLoader(dataset, batch_size=model_args.batch_size,
                      num_workers=model_args.workers, pin_memory=True,
                      shuffle=shuffle, drop_last=False)

VALID_SPURIOUS = [
    'TINT', # apply a fixed class-wise tinting (meant to not affect shape)
]
def add_spurious(ds, mode):
    assert mode in VALID_SPURIOUS

    loader = DataLoader(ds, batch_size=32, num_workers=1,
                        pin_memory=False, shuffle=False)

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs)
    ys = torch.cat(ys)

    colors = torch.tensor([(2, 1, 0), (1, 2, 0), (1, 1, 0),
                           (0, 2, 1), (0, 1, 2), (0, 1, 1),
                           (1, 0, 2), (2, 0, 1), (1, 0, 1),
                           (1, 1, 1)])

    colors = colors / torch.sum(colors + 0.0, dim=1, keepdim=True)

    xs_tint = (xs + colors[ys].unsqueeze(-1).unsqueeze(-1) / 3).clamp(0, 1)

    return folder.TensorDataset(xs_tint, ys)
