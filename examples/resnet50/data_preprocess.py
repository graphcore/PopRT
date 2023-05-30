# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import warnings

import numpy as np

try:
    import torchvision.transforms.functional as F

    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
    warnings.warn(f"Need torchvision and PIL packages for preprocessed data.")


def get_resnet50_preprocess_data(data_root, data_file):
    """Get resnet50 preprocessing data.

    :param data_root: the root of data
    :param data_file: file path for saving data information

    :return: {data, label}
    """

    count = 0
    f = open(data_file)
    file_names = f.readlines()
    for line in file_names:
        line_list = line.split()
        image_file = line_list[0]
        image_label = int(line_list[1])
        image_path = data_root + image_file
        if not os.path.exists(image_path):
            continue
        image_data = Image.open(image_path)
        image = F.resize(image_data, 256)
        image_crop = F.center_crop(image, 224)
        image_crop = np.array(image_crop)
        if len(image_crop.shape) != 3:
            if len(image_crop.shape) == 2:
                image_crop = np.array([image_crop, image_crop, image_crop])
                image_crop = np.transpose(image_crop, (1, 2, 0))
            else:
                continue
        image_norm = np.transpose(image_crop, (2, 0, 1))
        image_norm = image_norm[np.newaxis, :]
        img = np.ascontiguousarray(image_norm).astype(np.uint8)
        img = img[:, 0:3, :, :]
        if count == 0:
            image_batch = img
            label_batch = [image_label]
        else:
            image_batch = np.concatenate((image_batch, img), axis=0)
            label_batch.append(image_label)
        count += 1
    return image_batch, label_batch
