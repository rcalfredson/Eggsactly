"""PyTorch dataset for HDF5 files generated with `get_data.py`."""
import os
from random import random
import timeit
from typing import Optional

import cv2
import h5py
from PIL import Image
from sklearn.cluster import KMeans
import torch
from torchvision import transforms
import torchvision.transforms.functional as funcTrans
import numpy as np
from torch.utils.data import Dataset, DataLoader


def rotateImg(image, angle, center=None, scale=1.0, borderValue=(255, 255, 255), useInt=True):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=borderValue)

    return rotated/(255 if useInt else 1)


def background_color(img):
    # img = cv2.resize(img, (0, 0), fx=0.05, fy=0.05)
    # reshape = img.reshape(
    #     (img.shape[0] * img.shape[1], 3))
    
    # cluster = KMeans(n_clusters=1).fit(reshape)
    if type(img) != Image.Image:
        pil_img = Image.fromarray(img)
    else:
        pil_img = img
    return max(pil_img.getcolors(pil_img.size[0]*pil_img.size[1]))[1]
    # return tuple([int(255*val) for val in cluster.cluster_centers_[0]])

def padImagesBasedOnMaxSize(batch):
    maxHt, maxWidth = 0, 0
    for el in batch:
        img = el[0]
        if img.shape[1] > maxHt:
            maxHt = img.shape[1]
        if img.shape[-1] > maxWidth:
            maxWidth = img.shape[-1]
    resizedImgs = torch.zeros((len(batch), 3, maxHt, maxWidth), dtype=torch.float32)
    resizedLabels = torch.zeros((len(batch), 1, maxHt, maxWidth), dtype=torch.float32)
    # resizedImgs, resizedLabels = [], []
    for i, el in enumerate(batch):
        asymmetryCorrs = {'vp': 0, 'hp': 0}
        if el[0].shape[2] < maxWidth:
            dividend = (maxWidth - el[0].shape[2]) / 2
            asymmetryCorrs['hp'] = 1 if dividend % 1 > 0 else 0
            hp = int(dividend)
        else:
            hp = 0
        if el[0].shape[1] < maxHt:
            dividend = (maxHt - el[0].shape[1]) / 2
            asymmetryCorrs['vp'] = 1 if dividend % 1 > 0 else 0
            vp = int(dividend)
        else:
            vp = 0
        padding = (hp, hp + asymmetryCorrs['hp'], vp, vp + asymmetryCorrs['vp'])
        padded = np.multiply(el[0], 255).astype(np.uint8)
        # padded = torch.mul(el[0], 255)
        # padded = transforms.ToPILImage()(padded.T)
        bckgnd = background_color(padded)
        # padded = funcTrans.pad(padded, padding, bckgnd, 'constant')
        # print('shape before adding padding:', padded.shape)
        # print('shape after transpose:', padded.T.shape)
        # print('padding:', padding)
        # print('maxHt and maxWidth:', maxHt, maxWidth)
        padded = torch.nn.functional.pad(torch.Tensor(padded), padding, mode='constant', value=bckgnd)
        # print('shape after adding padding:', padded.shape)
        # resizedImgs.append(np.divide(np.array(padded).astype(np.float32), 255))
        resizedImgs[i] = torch.div(padded, 255.)
        # padded = transforms.ToPILImage()(el[1].T)
        # padded = funcTrans.pad(padded, padding, 0, 'constant')
        padded = torch.nn.functional.pad(torch.Tensor(el[1]), padding, mode='constant', value=0)
        # resizedLabels.append(np.expand_dims(np.array(padded), 0))
        # resizedLabels.append(np.array(padded))
        resizedLabels[i] = padded

    # preConversionTime = timeit.default_timer()
    # print('shapes and types of resized images:')
    # for img in resizedImgs:
    #     print(img.shape)
    #     print(type(img))
    # print('shapes of types of resized labels:')
    # for label in resizedLabels:
    #     print(label.shape)
    #     print(type(label))
    # imgTensor, labelTensor = torch.FloatTensor(resizedImgs), torch.FloatTensor(resizedLabels)
    # postConversionTime = timeit.default_timer() - preConversionTime
    # print('proportion of padding dims calc time to total:', postConversionTime / overallTotalTime)
    # print('shapes of final output tensors:', resizedImgs.shape, resizedLabels.shape)
    # print('content of one pair of resized images and labels:', resizedImgs[1], resizedLabels[1])
    return resizedImgs, resizedLabels

class MultiDimH5Dataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 horizontal_flip: float = 0.0,
                 vertical_flip: float = 0.0,
                 rand_rotate: bool = False,
                 rand_brightness = None):
        """
        Initialize flips probabilities and pointers to a HDF5 file.

        Args:
            dataset_path: a path to a HDF5 file
            horizontal_flip: the probability of applying horizontal flip
            vertical_flip: the probability of applying vertical flip
        """
        super(MultiDimH5Dataset, self).__init__()
        self.h5 = h5py.File(dataset_path, 'r')
        self.images = list(set([keyName.replace('_dots', '')
                                for keyName in self.h5.keys()]))
        self.labels = ["%s_dots" % imgName for imgName in self.images]
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rand_rotate = rand_rotate

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def __randrotate__(self, image, label):
        """Perform a random rotation on the image and labels."""
        if not self.rand_rotate or image.shape[0] != image.shape[1]:
            return (np.divide(image.astype(np.float32).T, 255), label)
        rotationAngle = np.random.uniform(-90, 90)
        backgnd_color = background_color(image)
        rotatedImg = rotateImg(image, rotationAngle,
            borderValue=backgnd_color)
        rotatedLabel = rotateImg(label.T, rotationAngle, borderValue=(0, 0, 0),
            useInt=False)
        rotatedLabel = np.expand_dims(rotatedLabel, axis=-1)
        rotatedImg = rotatedImg.astype(np.float32)

        return (rotatedImg.T, rotatedLabel.T)

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""
        # if both flips probabilities are zero return an image and a label
        if not (self.horizontal_flip or self.vertical_flip):
            return self.h5[self.images[index]][0], self.h5[self.labels[index]][0]

        # axis = 1 (vertical flip), axis = 2 (horizontal flip)
        axis_to_flip = []

        if random() < self.vertical_flip:
            axis_to_flip.append(1)

        if random() < self.horizontal_flip:
            axis_to_flip.append(2)

        image = np.flip(self.h5[self.images[index]][0], axis=axis_to_flip).copy()

        image = (255*image.T).astype(np.uint8)
        # cv2.imshow('preBright', image)
        # print('shape of image:', image.shape)
        # image = self.__randbrightness__(image)
        # cv2.imshow('postBright', image)
        # cv2.waitKey(0)

        final_results = self.__randrotate__(image, np.flip(self.h5[self.labels[index]][0],
            axis=axis_to_flip).copy())

        # cv2.imshow('imgFInal', final_results[0].T)
        # cv2.imshow('labelsFinal', final_results[1].T)
        # cv2.waitKey(0)

        return final_results


class H5Dataset(Dataset):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(self,
                 dataset_path: str,
                 horizontal_flip: float = 0.0,
                 vertical_flip: float = 0.0,
                 rand_rotate: bool = False,
                 rand_brightness: bool = False):
        """
        Initialize flips probabilities and pointers to a HDF5 file.

        Args:
            dataset_path: a path to a HDF5 file
            horizontal_flip: the probability of applying horizontal flip
            vertical_flip: the probability of applying vertical flip
        """
        super(H5Dataset, self).__init__()
        self.h5 = h5py.File(dataset_path, 'r')
        self.images = self.h5['images']
        self.labels = self.h5['labels']
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rand_rotate = rand_rotate
        self.rand_brightness = rand_brightness

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def __randrotate__(self, image, label):
        """Perform a random rotation on the image and labels."""
        if not self.rand_rotate:
            return (image.astype(np.float32).T, label)
        rotationAngle = np.random.uniform(-90, 90)
        backgnd_color = background_color(image)
        rotatedImg = rotateImg(image, rotationAngle,
            borderValue=backgnd_color)
        rotatedLabel = rotateImg(label.T, rotationAngle, borderValue=(0, 0, 0),
            useInt=False)
        rotatedLabel = np.expand_dims(rotatedLabel, axis=-1)
        rotatedImg = rotatedImg.astype(np.float32)

        return (rotatedImg.T, rotatedLabel.T)

    def __randbrightness__(self, image):
        value = np.random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""
        # if both flips probabilities are zero return an image and a label
        if not (self.horizontal_flip or self.vertical_flip):
            return self.images[index], self.labels[index]

        # axis = 1 (vertical flip), axis = 2 (horizontal flip)
        axis_to_flip = []

        if random() < self.vertical_flip:
            axis_to_flip.append(1)

        if random() < self.horizontal_flip:
            axis_to_flip.append(2)

        image = np.flip(self.images[index], axis=axis_to_flip).copy()

        image = (255*image.T).astype(np.uint8)
        # cv2.imshow('preBright', image)
        # print('shape of image:', image.shape)
        image = self.__randbrightness__(image)
        # cv2.imshow('postBright', image)
        # cv2.waitKey(0)

        final_results = self.__randrotate__(image, np.flip(self.labels[index],
            axis=axis_to_flip).copy())

        # cv2.imshow('imgFInal', final_results[0].T)
        # cv2.imshow('labelsFinal', final_results[1].T)
        # cv2.waitKey(0)

        return final_results


# --- PYTESTS --- #

def test_loader():
    """Test HDF5 dataloader with flips on and off."""
    run_batch(flip=False)
    run_batch(flip=True)


def run_batch(flip):
    """Sanity check for HDF5 dataloader checks for shapes and empty arrays."""
    # datasets to test loader on
    datasets = {
        'cell': (3, 256, 256),
        'mall': (3, 480, 640),
        'ucsd': (1, 160, 240)
    }

    # for each dataset check both training and validation HDF5
    # for each one check if shapes are right and arrays are not empty
    for dataset, size in datasets.items():
        for h5 in ('train.h5', 'valid.h5'):
            # create a loader in "all flips" or "no flips" mode
            data = H5Dataset(os.path.join(dataset, h5),
                             horizontal_flip=1.0 * flip,
                             vertical_flip=1.0 * flip)
            # create dataloader with few workers
            data_loader = DataLoader(data, batch_size=4, num_workers=4)

            # take one batch, check samples, and go to the next file
            for img, label in data_loader:
                # image batch shape (#workers, #channels, resolution)
                assert img.shape == (4, *size)
                # label batch shape (#workers, 1, resolution)
                assert label.shape == (4, 1, *size[1:])

                assert torch.sum(img) > 0
                assert torch.sum(label) > 0

                break
