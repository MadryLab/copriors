import torch
import torchvision
from torchvision import transforms
import numpy as np
import csv
from PIL import Image
import os

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
from robustness.tools import folder
import pandas as pd


def split_dataset(ds, split, folds=10, num_train_folds=2, num_val_folds=0):
    all_data_points = np.arange(len(ds))
    every_other = all_data_points[::folds]
    train_folds = num_train_folds
    val_folds = num_val_folds
    train_points = np.concatenate([every_other + i
                                    for i in range(0, train_folds)])
    if num_val_folds > 0:
        val_points = np.concatenate([every_other + i
                                       for i in range(train_folds,
                                                      train_folds + val_folds)])
    if folds - (train_folds + val_folds) > 0:
        unlabelled_points = np.concatenate([every_other + i
                                            for i in range(train_folds + val_folds,
                                                        folds)])
    if split == 'train':
        ds = torch.utils.data.Subset(ds, train_points)
    elif split.startswith('val'):
        if num_val_folds == 0:
            raise ValueError("Can't create a val set with 0 folds")
        ds = torch.utils.data.Subset(ds, val_points)
    else:
        if folds - (train_folds + val_folds) == 0:
            raise ValueError('no room for unlabelled points')
        ds = torch.utils.data.Subset(ds, unlabelled_points)
    return ds

class MetaDs(torch.utils.data.Dataset):
    def __init__(self, xs, ys, metas):
        self.xs = xs
        self.ys = ys
        self.metas = metas

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.metas[idx]

class Meta2Ds(torch.utils.data.Dataset):
    def __init__(self, ds, metas):
        self.ds = ds
        self.metas = metas

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, self.metas[idx]

class WaterBirds:
    CLASS_NAMES = ['land_bird', 'water_bird']
    MEAN = [0, 0, 0]
    STD = [1, 1, 1]
    SIZE = 96

    def __init__(self, data_path):
        self.data_path = data_path
        self.metadata = pd.read_csv(os.path.join(data_path, 'metadata.csv'))
        
    def get_dataset(self, split):
        assert split in ['train', 'test', 'unlabeled']
        split_idx_map = {'train': 0, 'unlabeled': 1, 'test': 2}
        split_idx = split_idx_map[split]
        df = self.metadata[self.metadata['split'] == split_idx]
        imgs = []
        ys = []
        metas = []
        iterator = tqdm(df.index)
        trans = transforms.Compose([
            transforms.Resize((WaterBirds.SIZE, WaterBirds.SIZE)),
            transforms.ToTensor(),
        ])

        for i in iterator:
            row = df.loc[i]
            filename = os.path.join(self.data_path, row['img_filename'])
            with Image.open(filename) as img:
                img_tensor = trans(img)
                imgs.append(img_tensor)
            ys.append(row['y'])
            agree = int(row['y'] == row['place'])
            metas.append({'agrees': agree})

        if split == 'test':
            return MetaDs(torch.stack(imgs), torch.tensor(ys), metas)
        else:
            return folder.TensorDataset(torch.stack(imgs), torch.tensor(ys))

class CelebA:
    """
    CelebA dataset
    """
    CLASS_NAMES = ['female', 'male']
    MEAN = [0, 0, 0]
    STD = [1, 1, 1,]
    SIZE = 96
    def __init__(self, data_path):
        self.data_path = data_path


    def get_dataset(self, split, unlabel_skew=True):
        assert split in ['train', 'val', 'test', 'unlabeled']
        trans = transforms.Compose([
            transforms.Resize((CelebA.SIZE, CelebA.SIZE)),
            transforms.ToTensor(),
        ])
        if split == 'test':
            ds = torchvision.datasets.CelebA(root=self.data_path, split='test', transform=trans)
        elif split == 'val':
            ds = torchvision.datasets.CelebA(root=self.data_path, split='valid', transform=trans)
        else:
            ds = torchvision.datasets.CelebA(root=self.data_path, split='train', transform=trans)
        attr_names = ds.attr_names
        attr_names_map = {a: i for i,a in enumerate(attr_names)}
        

        male_mask = ds.attr[:, attr_names_map['Male']] == 1
        female_mask = ds.attr[:, attr_names_map['Male']] == 0
        blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 1
        not_blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 0

        indices = torch.arange(len(ds))

        if split == 'train' or split == 'unlabeled':
            male_blond = indices[torch.logical_and(male_mask, blond_mask)]
            male_not_blond = indices[torch.logical_and(male_mask, not_blond_mask)]
            female_blond = indices[torch.logical_and(female_mask, blond_mask)]
            female_not_blond = indices[torch.logical_and(female_mask, not_blond_mask)]
            p_male_blond = len(male_blond)/float(len(indices))
            p_male_not_blond = len(male_not_blond)/float(len(indices))
            p_female_blond = len(female_blond)/float(len(indices))
            p_female_not_blond = len(female_not_blond)/float(len(indices))

            # training set must have 500 male_not_blond and 500 female_blond
            train_N = 500
            training_male_not_blond = male_not_blond[:train_N]
            training_female_blond = female_blond[:train_N]

            unlabeled_male_not_blond = male_not_blond[train_N:]
            unlabeled_female_blond = female_blond[train_N:]
            unlabeled_male_blond = male_blond
            unlabeled_female_not_blond = female_not_blond

            if unlabel_skew:
                # take 1000 from each category
                unlabeled_N = 1000
                unlabeled_male_not_blond = unlabeled_male_not_blond[:unlabeled_N]
                unlabeled_female_blond = unlabeled_female_blond[:unlabeled_N]
                unlabeled_male_blond = unlabeled_male_blond[:unlabeled_N]
                unlabeled_female_not_blond = unlabeled_female_not_blond[:unlabeled_N]
            else:
                total_N = 4000
                extra = total_N - int(p_male_not_blond*total_N) - int(p_female_blond*total_N) - int(p_male_blond*total_N) - int(p_female_not_blond*total_N)
                unlabeled_male_not_blond = unlabeled_male_not_blond[:int(p_male_not_blond*total_N)]
                unlabeled_female_blond = unlabeled_female_blond[:int(p_female_blond*total_N)]
                unlabeled_male_blond = unlabeled_male_blond[:int(p_male_blond*total_N)]
                unlabeled_female_not_blond = unlabeled_female_not_blond[:(int(p_female_not_blond*total_N) + extra)]
            
            train_indices = np.concatenate([training_male_not_blond, training_female_blond])
            unlabelled_indices = np.concatenate([unlabeled_male_not_blond, unlabeled_female_blond, unlabeled_male_blond, unlabeled_female_not_blond])
            for index in unlabelled_indices:
                assert index not in train_indices

            if split == 'train':
                indices = train_indices
            else:
                indices = unlabelled_indices 

        imgs = []
        ys = []
        metas = []
        for i in tqdm(indices):
            img, attr = ds[i]
            imgs.append(img)
            ys.append(attr[attr_names_map['Male']])
            if male_mask[i]:
                agree = False if blond_mask[i] else True
            else:
                agree = True if blond_mask[i] else False
            metas.append({'agrees': agree, 'blond': blond_mask[i], 'male': male_mask[i]})

        print(split, len(indices))
        if split == 'test':
            return MetaDs(torch.stack(imgs), torch.tensor(ys), metas)
        else:
            return folder.TensorDataset(torch.stack(imgs), torch.tensor(ys))

class CelebASkewed(CelebA):
    def get_dataset(self, split):
        return super().get_dataset(split, unlabel_skew=True)

class STL10:
    """
    STL-10 dataset
    """
    CLASS_NAMES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96
    def __init__(self, data_path):
        self.data_path = data_path


    def get_dataset(self, split):
        assert split in ['train', 'test', 'unlabeled']
        ds = torchvision.datasets.STL10(root=self.data_path, split=split, folds=None, download=False, transform=transforms.ToTensor())
        return ds


class STLSub10(STL10):
    def get_dataset(self, split):
        # 20% is labelled
        assert split in ['train', 'test', 'unlabeled', 'val']
        if split == 'test':
            ds = torchvision.datasets.STL10(root=self.data_path, split='test', folds=None, download=False, transform=transforms.ToTensor())
        else:
            ds = torchvision.datasets.STL10(root=self.data_path, split='train', folds=None, download=False, transform=transforms.ToTensor())
            ds = split_dataset(ds, split,
                               folds=10, num_train_folds=2, num_val_folds=1)
        return ds

class CIFAR:
    """
    CIFAR-10 dataset
    """
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    def __init__(self, data_path):
        self.data_path = data_path


    def get_dataset(self, split):
        assert split in ['train', 'test']
        ds = torchvision.datasets.CIFAR10(root=self.data_path, train=(split == 'train'), download=True, transform=transforms.ToTensor())
        return ds


class CIFARSmallSub(CIFAR):
    def get_dataset(self, split):
        # 1 % is labelled
        assert split in ['train', 'test', 'unlabeled', 'val']
        if split == 'test':
            ds = torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=transforms.ToTensor())
        else:
            ds = torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=transforms.ToTensor())
            ds = split_dataset(ds, split,
                               folds=100, num_train_folds=2, num_val_folds=10)
        return ds


COPRIOR_DATASETS = {
    'stl10': STL10,
    'STLSub10': STLSub10,
    'cifar': CIFAR,
    'cifarsmallsub': CIFARSmallSub,
    'celebaskewed': CelebASkewed,
}
