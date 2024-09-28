import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.n_ids = len(listdir(masks_dir))
        self.ids = listdir(masks_dir)
        self.ids.sort()
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        #logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load())
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load().numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class SyntheticDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, channels_sensors=[1, 3], type_disruptions=[None, None],
                 prop_disruptions=[0., 0.], level_disruptions = [0, 0], sensor_ids_list = [0,1]):
                 
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
        
        self.images_dir = [f"{images_dir}/images_{sensor_id}/" for sensor_id in sensor_ids_list]
        self.channels_sensors = channels_sensors
        self.type_disruptions = type_disruptions
        self.prop_disruptions = prop_disruptions  # 0 no disruptions , 1 all disruptions
        self.level_disruptions = level_disruptions
        self.sensor_ids_list = sensor_ids_list
        
        self.disturb_parameter_levels = {'blur': {'ksize': [(10,10),(20,20),(40,40),(80,80),(160,160)]},
                             'snp_delta': {'delta': [16,32,64,128,256], 'grain_size': [10,10,10,10,10]},
                             'snp_grain': {'delta': [64,64,64,64,64], 'grain_size': [1,5,10,20,40]},
                             'delete_pixel': {'ratio_del_pixel': [0.2,0.4,0.6,0.8,1.0]}}

        if len(self.type_disruptions) > len(self.sensor_ids_list):
            self.type_disruptions = [self.type_disruptions[i] for i in self.sensor_ids_list]
        if len(self.prop_disruptions) > len(self.sensor_ids_list):
            self.prop_disruptions = [self.prop_disruptions[i] for i in self.sensor_ids_list]

        assert len(self.type_disruptions) == len(self.sensor_ids_list)
        assert len(self.prop_disruptions) == len(self.sensor_ids_list)

    def __getitem__(self, idx):
        name = self.ids[idx]

        mask_file = f"{self.masks_dir}/{name}"
        img_files = [f"{images_dir}/{name}" for images_dir in self.images_dir]

        mask = self.load(mask_file, channels=1)
        imgs = [self.load(img_file, c_s) for img_file, c_s in zip(img_files, self.channels_sensors)]


        for i, type_dist in enumerate(self.type_disruptions):
            if np.random.uniform() < self.prop_disruptions[i]:
                imgs[i] = self.disturb_img(imgs[i], level=self.level_disruptions[i], type_dist=type_dist)

        assert all([i.size / c_s == mask.size for c_s, i in zip(self.channels_sensors, imgs)]), \
            f'Image and mask {name} should be the same size, but are '\
            f'{[i.size / c_s for c_s, i in zip(self.channels_sensors, imgs)]} and {mask.size}'

        mask = self.preprocess(mask, self.scale, is_mask=True)
        imgs = [self.preprocess(img, self.scale, is_mask=False) for img in imgs]
        imgs = np.concatenate(imgs)
        return {
            'image': torch.as_tensor(imgs.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def disturb_img(self, img, level = 0, type_dist="none"):
        if level == 0:
            return img
        if type_dist == "blur":
            return cv2.blur(img, ksize=self.disturb_parameter_levels[type_dist]['ksize'][level-1])
        elif 'snp' in type_dist:
            k = saltNPepper(img, delta=self.disturb_parameter_levels[type_dist]['delta'][level-1],
                                 grain_size=self.disturb_parameter_levels[type_dist]['grain_size'][level-1])
            return k
        elif type_dist == "delete_pixel":
            return delete_pixel(img, ratio_del_pixel=self.disturb_parameter_levels[type_dist]['ratio_del_pixel'][level-1])
        elif type_dist == "none":
            return img
        else:
            print("Warining: Type disturtion not found. Using None.")
            return img

    @staticmethod
    def load(filename, channels=1):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return torch.load(filename).numpy()
        else:
            if channels == 1:
                return np.expand_dims(cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE), 2)
            if channels == 3:
                return cv2.imread(filename, flags=cv2.IMREAD_COLOR)

    @staticmethod
    def preprocess(img, scale, is_mask):
        if img.ndim == 3:
            w, h, c = img.shape
        else:
            w, h = img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)
        img_ndarray = np.asarray(img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255
        else:
            img_ndarray = np.int32(img_ndarray / 255)

        return img_ndarray


def saltNPepper(img, delta=128, grain_size=4):
    h, w, c = img.shape
    snp_h = int(h / grain_size)
    snp_w = int(w / grain_size)
    snp = np.random.randint(-delta, delta, size=[snp_h, snp_w,c])
    snp = cv2.resize(snp, (w, h), interpolation=cv2.INTER_NEAREST).reshape(w, h, c)
    img = img.astype(np.int32) + snp
    return np.clip(img, 0, 255).astype(np.uint8)

def delete_pixel(img, ratio_del_pixel):
    n = np.prod(img.shape[0:2])
    del_mask = np.array([True] * int(np.round(n * ratio_del_pixel)) + [False] * int(np.round(n * (1-ratio_del_pixel))))
    np.random.shuffle(del_mask)
    del_mask = del_mask.reshape(img.shape[0:2])
    img[del_mask,:] = 0
    return img

if __name__ == "__main__":
    dataset = SyntheticDataset("../../SyntheticDummyDataset_LightAndColor/data/test/",
                               "../../SyntheticDummyDataset_LightAndColor/data/test/labels/",
                               type_disruptions=["None", "snp"],
                               prop_disruptions=[1.],
                               sensor_ids_list=[0])
    items = dataset.__getitem__(10)
    img = items["image"]
    mask = items["mask"]
    print(img.shape)
