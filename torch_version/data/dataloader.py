import os
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


# class_mapping is used to encode label into one hot mapping.
class_mapping = {
    "R": 0,
    "C": 1,
    "U": 2,
    "Q": 3,
    "J": 4,
    "L": 5,
    "RA": 6,
    "D": 7,
    "RN": 8,
    "TP": 9,
    "IC": 10,
    "P": 11,
    "CR": 12,
    "M": 13,
    "BTN": 14,
    "FB": 15,
    "CRA": 16,
    "SW": 17,
    "T": 18,
    "F": 19,
    "V": 20,
    "LED": 21,
    "S": 22,
    "QA": 23,
    "JP": 24,
}

# color_values is used to encode mask into one hot mapping. following color needs to be updated based on the color used while creating masks
color_values = {
    0: (255, 0, 0),
    1: (255, 255, 0),
    2: (0, 234, 255),
    3: (170, 0, 255),
    4: (255, 127, 0),
    5: (191, 255, 0),
    6: (0, 149, 255),
    7: (106, 255, 0),
    8: (0, 64, 255),
    9: (237, 185, 185),
    10: (185, 215, 237),
    11: (231, 233, 185),
    12: (220, 185, 237),
    13: (185, 237, 224),
    14: (143, 35, 35),
    15: (35, 98, 143),
    16: (143, 106, 35),
    17: (107, 35, 143),
    18: (79, 143, 35),
    19: (115, 115, 115),
    20: (204, 204, 204),
    21: (245, 130, 48),
    22: (220, 190, 255),
    23: (170, 255, 195),
    24: (255, 250, 200),
    25: (0, 0, 0)
}


def get_paths(opt):
    if opt["type"] == "Segmentation":
        images = sorted(
            [os.path.join(opt["data_images"], img) for img in os.listdir(opt["data_images"])])
        masks = sorted(
            [os.path.join(opt["data_masks"], msk) for msk in os.listdir(opt["data_masks"])])
        return images, masks

class SegDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.images_dir, self.masks_dir = get_paths(opt)


    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        image_path = self.images_dir[idx]
        mask_path = self.masks_dir[idx]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.opt["img_size_h"], self.opt["img_size_w"]))
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = Image.open(mask_path).convert("RGB")
        mask = mask.resize((self.opt["img_size_h"], self.opt["img_size_w"]))
        mask_ = ToTensor()(mask)
        mask = np.array(mask)
        one_hot_map = []
        for colour in list(color_values.values()):
            class_map = np.all(mask == colour, axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = np.stack(one_hot_map, axis=-1)
        one_hot_map = torch.from_numpy(one_hot_map).permute(2, 0, 1).float()

        return {'input': image, 'gt': one_hot_map, 'gt_mask': mask_}


def get_data(opt):
    train_opt = opt["datasets"]["train"]
    val_opt = opt["datasets"]["val"]
    train_images, train_targets = get_paths(train_opt)
    val_images, val_targets = get_paths(val_opt)

    train_dataset = SegDataset(train_opt)
    val_dataset = SegDataset(val_opt)

    train_dataloader = DataLoader(train_dataset, batch_size=train_opt["batch_size"],
                                  shuffle=train_opt["use_shuffle"], num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=val_opt["batch_size"],
                                shuffle=False, num_workers=8)

    return train_dataloader, val_dataloader


