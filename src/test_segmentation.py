import argparse
import logging
import time

import tensorflow as tf
import numpy as np
import os
from PIL import Image

from utils import dict2str, parse, get_msg
from models import get_model
from data import get_data
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

from models.loss import dice_coef

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

def parse_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        default="cfs/pscn_seg.yml",
        help="Path to option YAML file."
    )
    parser.add_argument(
        "-weights",
        type=str,
        default="../checkpoints/model_200.h5",
        help="Path to the trained model weights."
    )
    parser.add_argument(
        "-output",
        type=str,
        default="results/orig_E200_image0",
        help="Directory to save prediction images."
    )
    args = parser.parse_args()

    opt = parse(args.opt)
    opt["test"] = {
        "weights_path": args.weights,
        "output_dir": args.output
    }
    return opt


def convert_to_color_image(output, color_values):
    B, H, W, C, = output.shape
    color_image = np.zeros((B, H, W, 3), dtype=np.uint8)

    for b in range(B):
        # print(f"batch {b}")
        for h in range(H):
            for w in range(W):
                color_index = np.argmax(output[b, h, w])
                color_image[b, h, w] = color_values[color_index]

    return color_image


def init_test_log(opt):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt["test"]["output_dir"], "test_log.txt"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(get_msg())
    logger.info(dict2str(opt))


def test_model(opt):
    logger = logging.getLogger(__name__)

    # Get test dataset
    _, test_dataset = get_data(opt)
    logger.info(f"Found {len(test_dataset)} batches for testing")

    # Get model
    model = get_model(opt)
    model.load_weights(opt["test"]["weights_path"])
    logger.info(f"Loaded weights from {opt['test']['weights_path']}")

    # Create output directory
    os.makedirs(opt["test"]["output_dir"], exist_ok=True)

    # eval_results = model.evaluate(test_dataset)
    # y_pred = model.predict(test_dataset)
    dice_co = []
    # Test and save predictions
    for i, (x, hot_map, gt) in enumerate(test_dataset):
        y_pred = model.predict(x)
        hm = hot_map.numpy()
        dc = dice_coef(y_true=hot_map, y_pred=y_pred)
        dice_co.append(dc.numpy())
        print(dc.numpy())
        a = time.time()
        # print(f"pred {i} done, at {a}")
        color_image = convert_to_color_image(y_pred, color_values)
        b = time.time()-a
        print(f"color {i} done, using {b}s")
        for j in range(color_image.shape[0]):
            output_path = os.path.join(opt["test"]["output_dir"], f"_{i}_pred.png")
            # gt_path = os.path.join(opt["test"]["output_dir"], f"_{i}_{j}_gt.png")
            color_image_pil = Image.fromarray(color_image[j])
            color_image_pil.save(output_path)
            # gt = tf.cast(gt, tf.uint8)
            # gt = gt.numpy()
            # gt_pil = Image.fromarray(gt[j])
            # gt_pil.save(gt_path)
    print(np.mean(dice_co))

    logger.info(f"Saved prediction images to {opt['test']['output_dir']}")


def main():
    opt = parse_test_config()
    init_test_log(opt)
    test_model(opt)


if __name__ == "__main__":
    main()