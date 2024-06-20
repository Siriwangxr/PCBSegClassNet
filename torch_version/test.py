import os
import sys
from PIL import Image

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

import argparse
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from utils.options import dict2str, parse, get_msg
from models.blocks import PCBModel
from data.dataloader import get_data
from models.loss import DISLoss, dice_coef, jacard_coef
from torchmetrics import Dice
from skimage.metrics import structural_similarity


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


def convert_to_color_image(output, color_values):
    B, C, H, W = output.shape
    color_image = np.zeros((B, H, W, 3), dtype=np.uint8)

    for b in range(B):
        for c in range(C):
            color = color_values[c]
            mask = output[b, c] == 1
            color_image[b][mask] = color

    return color_image


def parse_config():
    """
    Helper function to parse config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        default='configs/pscn_seg.yml',
        help="Path to option YAML file."
    )
    parser.add_argument("-epoch",
                        type=int,
                        default=1,
                        help="number of epochs.")
    args = parser.parse_args()

    opt = parse(args.opt)
    opt["train"]["total_epochs"] = args.epoch
    return opt


def test(model, test_loader, device, opt):
    model.eval()

    dice = Dice(num_classes=26)
    total_dice_score = 0
    total_ssim = 0

    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            # Move the input data to the device
            test_data['input'] = test_data['input'].to(device)
            test_data['gt'] = test_data['gt'].to(device)

            # Forward pass
            y_pred = model(test_data['input'])
            dice_score = torch.mean(dice_coef(y_pred, test_data['gt']))
            total_dice_score += dice_score.item()
            iou_score = torch.mean(jacard_coef(y_pred, test_data['gt']))
            total_ssim += iou_score.item()


            # Get the predicted class indices
            y_pred_indices = torch.argmax(y_pred, dim=1)
            y_pred_indices = y_pred_indices.cpu().numpy()
            gt_indices = test_data['gt'].cpu().numpy().astype(np.uint8)

            # Create a color map
            num_classes = len(color_values)
            color_map = np.zeros((num_classes, 3), dtype=np.uint8)
            for class_index, color in color_values.items():
                color_map[class_index] = color

            # Map the class indices to RGB values
            segmentation_rgb = color_map[y_pred_indices]
            gt_rgb = color_map[gt_indices]
            # ssim = structural_similarity(segmentation_rgb[0], test_data['gt_mask'].squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8), multichannel=True)

            # Accumulate the SSIM value


            # Convert the RGB array to an image and save
            segmentation_image = Image.fromarray(segmentation_rgb[0])
            save_path = opt['image_save']
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            segmentation_image.save(f'{save_path}/_{batch_idx}_segmentation_result.png')
            save_image(test_data['gt_mask'], f'{save_path}/_{batch_idx}_gt_result.png')
    num_batches = len(test_loader)
    average_dice_score = total_dice_score / num_batches
    average_ssim = total_ssim / num_batches
    print(f"Average Dice Coefficient: {average_dice_score:.4f}")
    print(f"Average SSIM: {average_ssim:.4f}")



def main():
    opt = parse_config()

    gpu_ids = opt['gpu_ids']
    if gpu_ids:
        device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Load the trained model
    model = PCBModel(opt['train']['num_classes']).to(device)
    checkpoint = torch.load('ckpt/PCBSegNet_epoch_200.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Dataset
    _, test_loader = get_data(opt)

    # Test the model
    test(model, test_loader, device, opt)


if __name__ == '__main__':
    main()