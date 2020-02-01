from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence


def resize_img(img, min_side=900):
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                min_side - new_w) / 2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目

    return pad_img


class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir, source_noise_model, target_noise_model, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path = random.choice(self.image_paths)
            image = cv2.imread(str(image_path))
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                # crop a random patch in src image, like random crop
                clean_patch = image[i:i + image_size, j:j + image_size]
                x[sample_id] = self.source_noise_model(clean_patch)
                y[sample_id] = self.target_noise_model(clean_patch)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y
            else:
                image = resize_img(image, min_side=image_size)

                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                # crop a random patch in src image, like random crop
                clean_patch = image[i:i + image_size, j:j + image_size]
                x[sample_id] = self.source_noise_model(clean_patch)
                y[sample_id] = self.target_noise_model(clean_patch)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y

class ValGenerator(Sequence):
    def __init__(self, image_dir, val_noise_model):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_path in image_paths:
            y = cv2.imread(str(image_path))
            h, w, _ = y.shape
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            x = val_noise_model(y)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
