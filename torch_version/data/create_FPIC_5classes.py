from glob import glob
import argparse
import sys
import ast
import os
from skimage.transform import resize
import albumentations as A
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2

color_values = {
    "R": 1,
    "C": 2,
    "U": 3,
    "L": 4,
    "IC": 3,
    "T": 5,
}

def resize_with_aspect_ratio(image, size, padding_color=(0, 0, 0)):
    h, w = image.shape[:2]
    sh, sw = size

    aspect = w / h
    if aspect > 1:  # wider
        new_w = sw
        new_h = int(sw / aspect)
        pad_vert = (sh - new_h) / 2
        pad_top = int(pad_vert)
        pad_bot = sh - new_h - pad_top
        pad_left = 0
        pad_right = 0
    else:  # taller
        new_h = sh
        new_w = int(sh * aspect)
        pad_horz = (sw - new_w) / 2
        pad_left = int(pad_horz)
        pad_right = sw - new_w - pad_left
        pad_top = 0
        pad_bot = 0

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padding_color)
    return padded_image

def prepare_data(source_image_dir,
                 source_annotation_dir,
                 dest_images_dir,
                 dest_masks_sir,
                 crops_dest_dir,
                 ):
    """
    Helper function which creates masks and croops
    Args:
        source_image_dir: image directory containing input images
        source_annotation_dir: annotation directory containing csv annotations
        dest_images_dir: destination directory for storing images
        dest_masks_sir: destination directory for storing masks
        dest_crops_dir: destinations directory for storing crops
        models: super resolution models
    """
    annotations_list = glob(os.path.join(source_annotation_dir, "*.csv"))
    count = 0
    cnt = 0
    # transform = A.Compose([
    #     A.augmentations.transforms.CLAHE(clip_limit=4.0,
    #                                      tile_grid_size=(8, 8),
    #                                      always_apply=False,
    #                                      p=1.0)
    # ])
    with tqdm(total=len(annotations_list)) as pbar:
        for annotation in annotations_list:
            df = pd.read_csv(annotation)
            # checking if atleast 1 designation is present in annotation
            if df["Designator"].isna().sum() != df.shape[0]:
                image_name = list(df["Image File"].unique())
                if os.path.exists(os.path.join(source_image_dir, image_name[0])):
                    img = cv2.imread(os.path.join(source_image_dir, image_name[0]))
                    vertices_list = list(df["Vertices"])
                    designator_list = list(df["Designator"])
                    for (anote, cat) in zip(vertices_list, designator_list):
                        if cat in color_values:
                            class_index = color_values[cat]
                        else:
                            continue
                        try:
                            pts = np.array(ast.literal_eval(anote))[0].reshape((-1, 1, 2))
                        except:
                            continue
                        # mask = cv2.polylines(mask, [pts], True, color_code, 2)
                        # mask = cv2.fillPoly(mask, [pts], color=color_code)
                        # create crops
                        mask_copy = np.zeros(shape=img.shape[:2], dtype=np.uint8)
                        mask_copy = cv2.polylines(mask_copy, [pts], True, 255, 2)
                        mask_copy = cv2.fillPoly(mask_copy, [pts], color=255)
                        # mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
                        contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)
                        x,y,w,h = cv2.boundingRect(contours[0])

                        # Add a 5% margin to the bounding rectangle
                        margin = 0.05
                        x_margin = int(w * margin)
                        y_margin = int(h * margin)
                        x = max(0, x - x_margin)
                        y = max(0, y - y_margin)
                        w = min(img.shape[1] - x, w + 2 * x_margin)
                        h = min(img.shape[0] - y, h + 2 * y_margin)

                        crop_img = img[y:y+h, x:x+w]
                        crop = resize_with_aspect_ratio(crop_img, (512, 512))

                        # comp_mask = np.zeros((img.shape[0], img.shape[1], 6), dtype=np.uint8)
                        # comp_mask[:, :, class_index] = mask_copy
                        # comp_mask_resized = np.zeros((512, 512, 6), dtype=np.uint8)
                        # for i in range(6):
                        #     comp_mask_resized[:, :, i] = resize_with_aspect_ratio(comp_mask[:, :, i], (512, 512),
                        #                                                           padding_color=0)
                        mask = mask_copy[y:y+h, x:x+w]
                        mask = resize_with_aspect_ratio(mask, (512, 512))
                        #adding a threshold 128 to mask
                        mask = (mask > 128).astype(np.float32)
                        comp_mask = np.zeros((512, 512, 6), dtype=np.float32)
                        comp_mask[:, :, class_index] = mask
                        comp_mask[:, :, 0] = 1. - mask

                        if not os.path.exists(os.path.join(crops_dest_dir, cat)):
                            os.makedirs(os.path.join(crops_dest_dir, cat))
                        cv2.imwrite(os.path.join(crops_dest_dir, cat, f"image_{cnt}.png"), cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        if not os.path.exists(os.path.join(dest_masks_sir, cat)):
                            os.makedirs(os.path.join(dest_masks_sir, cat))
                        np.save(os.path.join(dest_masks_sir, cat, f"mask_{cnt}.npy"), comp_mask)
                        visual_mask_dir = "../../../dataset/FPIC/segmentation/masks_visual_2"
                        if not os.path.exists(os.path.join(visual_mask_dir, cat)):
                            os.makedirs(os.path.join(visual_mask_dir, cat))
                        cv2.imwrite(os.path.join(visual_mask_dir, cat, f"mask_{cnt}.png"), mask)

                        cnt += 1


                    count+=1
                    pbar.update(1)


def main(source_image_dir,
         source_annotation_dir,
         dest_images_dir,
         dest_masks_sir,
         dest_crops_dir,
         ):
    """
    main function which creates mask
    Args:
        source_image_dir: image directory containing input images
        source_annotation_dir: annotation directory containing csv annotations
        dest_images_dir: destination directory for storing images
        dest_masks_sir: destination directory for storing masks
        dest_crops_dir: destinations directory for storing crops
        models: super resolution models # not used
    """
    # create directories if not exist
    if not os.path.exists(dest_images_dir):
        os.makedirs(dest_images_dir)

    if not os.path.exists(dest_masks_sir):
        os.makedirs(dest_masks_sir)

    if not os.path.exists(dest_crops_dir):
        os.makedirs(dest_crops_dir)

    prepare_data(source_image_dir,
                 source_annotation_dir,
                 dest_images_dir,
                 dest_masks_sir,
                 dest_crops_dir,
                 )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PCBSegClassNet')
    parser.add_argument('-i',
                        "--images_dir",
                        type=str,
                        default="../../../dataset/FPIC/pcb_image",
                        help="The path of directory containing input images")
    parser.add_argument('-a',
                        '--annotations_dir',
                        type=str,
                        default="../../../dataset/FPIC/smd_annotation",
                        help="The path of directory containing annotations")
    parser.add_argument('-id',
                        '--images_dest_dir',
                        type=str,
                        default="../../../dataset/FPIC/segmentation/images",
                        help="The path of destination directory where images needs to be stored")
    parser.add_argument('-ad',
                        '--annotations_dest_dir',
                        type=str,
                        default="../../../dataset/FPIC/segmentation/masks",
                        help="The path of destination directory where masks needs to be stored")
    parser.add_argument('-cd',
                        '--crops_dest_dir',
                        type=str,
                        default="../../../dataset/FPIC/segmentation/crops_2",
                        help="The path of destination directory where crops needs to be stored")
    args = parser.parse_args()

    main(source_image_dir = args.images_dir,
         source_annotation_dir = args.annotations_dir,
         dest_images_dir = args.images_dest_dir,
         dest_masks_sir = args.annotations_dest_dir,
         dest_crops_dir = args.crops_dest_dir,
         )
