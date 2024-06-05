# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Network definition for PCBSegNet and PCBClassNet
"""

import tensorflow as tf
import models

from . import get_encoder, get_decoder

class PCBSegNet:
    """
    Main class for segmentation models creation
    """
    def __init__(self, opt):
        """
        Args:
            opt: training options
        """
        self.model_type = opt["model_type"]
        self.image_height = opt["datasets"]["train"]["img_size_h"]
        self.image_width = opt["datasets"]["train"]["img_size_w"]
        self.num_classes = opt["train"]["num_classes"] + 1

    def build(self):
        """
        build encoder, decoder and final models
        """
        encoder, learning_layer1, learning_layer2 = get_encoder(self.image_height, self.image_width)
        model = get_decoder(encoder, learning_layer1, learning_layer2, self.num_classes)
        print(model.summary())
        return model


def get_model(opt):
    if opt["model_type"] == "SegmentationModel":
        seg_model = PCBSegNet(opt)
        model = seg_model.build()
        optimizer = getattr(tf.keras.optimizers, opt["train"]["optim"]["type"])(
            learning_rate=opt["train"]["optim"]["lr"],
            beta_1=opt["train"]["optim"]["betas"][0],
            beta_2=opt["train"]["optim"]["betas"][1],
        )
        loss = getattr(models, opt["train"]["loss"]["type"])
        metrics = [
            getattr(models, opt["train"]["metric"][item]["type"])
            for item in opt["train"]["metric"]
        ]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
