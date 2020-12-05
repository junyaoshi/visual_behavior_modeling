import argparse
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import time
import glob
import numpy as np
from tqdm import tqdm


class assembly_dataset(Dataset):
    "Assembly Dataset"

    def __init__(self, data_dirs, window_size=2):
        """
           :param data_dir: directory where assembly data is stored
           :param window_size: size of sliding window
           :param train_ratio: the percentage of data used for training
           :param img_dim: dimensions of the image

            * at the moment, only using the first camera
        """
        self.data_dirs = data_dirs
        self.window_size = window_size
        self.n_cam = 0

        num_imgs_total = 0
        self.data_bins = []  # list of bins, where each dir's # of data is the bin diff
        print('Loading data into dataset...')
        for data_dir_idx in tqdm(range(len(self.data_dirs))):
            data_dir = self.data_dirs[data_dir_idx]
            num_imgs = len(glob.glob(os.path.join(data_dir, '{}'.format(self.n_cam), '*.png')))
            num_imgs_window = num_imgs - self.window_size
            num_imgs_total += num_imgs_window
            self.data_bins.append(num_imgs_total)

        self.data_len = num_imgs_total

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_dir_idx = np.digitize(idx, self.data_bins)  # find idx of bin the given idx belongs to
        data_dir = self.data_dirs[data_dir_idx]

        if data_dir_idx == 0:
            start_frame_idx = idx
        else:
            start_frame_idx = idx - self.data_bins[data_dir_idx - 1]
        target_frame_idx = start_frame_idx + self.window_size

        data = []
        # populate sliding window with images
        for window_idx in range(start_frame_idx, target_frame_idx):
            img_path = os.path.join(data_dir, '{}'.format(self.n_cam), 'rgb_{}.png'.format(window_idx))
            im = np.array(Image.open(img_path))
            data.append(im)

        data = np.dstack(data) / 255.0

        target_img_path = os.path.join(data_dir, '{}'.format(self.n_cam), 'rgb_{}.png'.format(target_frame_idx))
        target = np.array(Image.open(target_img_path)) / 255.0

        return data, target

    def get_labels(self):
        print('Loading data sequence labels...')
        labels_dict = {}
        for data_dir_idx in tqdm(range(len(self.data_dirs))):
            data_dir = self.data_dirs[data_dir_idx]
            labels = np.load(os.path.join(data_dir, 'labels.npy'))
            labels_dict[data_dir] = labels
        return labels_dict


def load_data(data_dir, window_size=2, train_ratio=0.8, img_dim=(128, 128, 3)):
    """
    :param data_dir: directory where assembly data is stored
    :param window_size: size of sliding window
    :param train_ratio: the percentage of data used for training
    :param img_dim: dimensions of the image

    * at the moment, only using the first camera

    return:
        train_data, test_data, train_target, test_target: normalized training and testing X, y
        train_seq_inds, test_seq_inds: the indices of training and testing sequences
    """

    print('Begin loading data from {}...'.format(data_dir))

    h, w, ch = img_dim

    # train-test split
    seq_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    num_seq = len(seq_dirs)
    split_idx = int(round(train_ratio * num_seq))
    np.random.shuffle(seq_dirs)
    train_seqs = seq_dirs[:split_idx]
    test_seqs = seq_dirs[split_idx:]
    print('Split data into {} training sequences and {} testing sequences.'.format(len(train_seqs), len(test_seqs)))

    # load training data
    print('Generating training data...')
    train_data = np.empty((0, h, w, ch * window_size), np.float)
    train_target = np.empty((0, h, w, ch), np.float)
    train_seq_inds = []
    for seq_ind in tqdm(range(len(train_seqs))):
        seq_dir = train_seqs[seq_ind]
        n_seq = int(seq_dir[-1])
        train_seq_inds.append(n_seq)

        n_cam = 0

        seq_data = []
        seq_target = []

        cam_dir = os.path.join(seq_dir, '{}'.format(n_cam))
        num_imgs = len(glob.glob(os.path.join(cam_dir, '*.png')))
        sliding_window = deque()
        left = 0
        right = window_size - 1
        target_idx = right + 1
        target_img_path = os.path.join(cam_dir, 'rgb_{}.png'.format(target_idx))

        # populate sliding window with images
        for window_idx in range(left, right + 1):
            img_path = os.path.join(cam_dir, 'rgb_{}.png'.format(window_idx))
            im = np.array(Image.open(img_path)) / 255.0
            sliding_window.append(im)

        while target_idx < num_imgs:
            data_imgs = np.dstack(sliding_window)
            target_img = np.array(Image.open(target_img_path)) / 255.0

            seq_data.append(data_imgs)
            seq_target.append(target_img)

            left += 1
            right += 1
            target_idx += 1
            target_img_path = os.path.join(cam_dir, 'rgb_{}.png'.format(target_idx))

            sliding_window.popleft()
            sliding_window.append(target_img)

        seq_data = np.array(seq_data)
        seq_target = np.array(seq_target)
        train_data = np.vstack([train_data, seq_data])
        train_target = np.vstack([train_target, seq_target])

    # load testing data
    print('Generating testing data...')
    test_data = np.empty((0, h, w, ch * window_size), np.float)
    test_target = np.empty((0, h, w, ch), np.float)
    test_seq_inds = []
    for seq_ind in tqdm(range(len(test_seqs))):
        seq_dir = test_seqs[seq_ind]
        n_seq = int(seq_dir[-1])
        test_seq_inds.append(n_seq)

        n_cam = 0

        seq_data = []
        seq_target = []

        cam_dir = os.path.join(seq_dir, '{}'.format(n_cam))
        num_imgs = len(glob.glob(os.path.join(cam_dir, '*.png')))
        sliding_window = deque()
        left = 0
        right = window_size - 1
        target_idx = right + 1
        target_img_path = os.path.join(cam_dir, 'rgb_{}.png'.format(target_idx))

        # populate sliding window with images
        for window_idx in range(left, right + 1):
            img_path = os.path.join(cam_dir, 'rgb_{}.png'.format(window_idx))
            im = np.array(Image.open(img_path)) / 255.0
            sliding_window.append(im)

        while target_idx < num_imgs:
            data_imgs = np.dstack(sliding_window)
            target_img = np.array(Image.open(target_img_path)) / 255.0

            seq_data.append(data_imgs)
            seq_target.append(target_img)

            left += 1
            right += 1
            target_idx += 1
            target_img_path = os.path.join(cam_dir, 'rgb_{}.png'.format(target_idx))

            sliding_window.popleft()
            sliding_window.append(target_img)

        seq_data = np.array(seq_data)
        seq_target = np.array(seq_target)
        test_data = np.vstack([test_data, seq_data])
        test_target = np.vstack([test_target, seq_target])

    train_seq_inds = np.array(train_seq_inds)
    test_seq_inds = np.array(test_seq_inds)

    return train_data, test_data, train_target, test_target, train_seq_inds, test_seq_inds