import os
import sys

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

import argparse
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.options import dict2str, parse, get_msg
from models.blocks import PCBModel
from data.dataloader import get_data
from models.loss import DISLoss


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


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def main():
    opt = parse_config()
    max_epoch = opt["train"]["max_epoch"]

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Specify the device (GPU0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_loader, _ = get_data(opt)

    # Initialize the model and move it to the device
    model = PCBModel(opt['train']['num_classes']).to(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['train']['optim']['lr'], betas=(0.9, 0.9))

    # Create a SummaryWriter for TensorBoard logging
    writer = SummaryWriter(log_dir='logs/{}'.format(opt['name']))

    current_epoch = 0
    with tqdm(total=max_epoch, unit='epoch') as pbar:
        while current_epoch < max_epoch+1:
            current_epoch += 1
            model.train()
            for batch_idx, train_data in enumerate(train_loader):
                # Move the input data to the device
                train_data['input'] = train_data['input'].to(device)
                train_data['gt'] = train_data['gt'].to(device)


                y_pred = model(train_data['input'])
                y_true = train_data['gt']
                loss = DISLoss(y_true, y_pred)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Log the loss to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), current_epoch * len(train_loader) + batch_idx)
                # color_images = convert_to_color_image(y_pred.detach().cpu().numpy(), color_values)
                # writer.add_image('Color images', torch.from_numpy(color_images[0]).permute(2, 0, 1), current_epoch)
                # gt_masks = convert_to_color_image(y_true.detach().cpu().numpy(), color_values)
                # writer.add_image('GT_Mask', torch.from_numpy(gt_masks[0]).permute(2, 0, 1), current_epoch)

            if current_epoch % 50 == 0:
                checkpoint = {
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }
                save_checkpoint(checkpoint, filename=f"models/PCBSegNet_epoch_{current_epoch}.pth.tar")

            # Update the progress bar
            pbar.update(1)

    # Close the SummaryWriter
    writer.close()

if __name__ == "__main__":
    main()
