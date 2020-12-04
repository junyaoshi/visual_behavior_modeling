from __future__ import print_function
import argparse
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import time
import glob
import shutil
import itertools
import numpy as np
from utils import Bar, Logger, AverageMeter, regression_accuracy, mkdir_p, savefig
from new_models.assembly_model import TrajPredictor

eval_dir = 'assembly_evaluation'
window_size = 2
num_key_frames = 4
img_dim = (128, 128, 3)
model_path = os.path.join('checkpoint', 'model_best.pth.tar')
model = TrajPredictor(window_size * 3)
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()


def load_data(data_dir, window_size=window_size, img_dim=img_dim):
    """
    :param data_dir: directory where assembly data is stored
    :param window_size: size of sliding window
    :param img_dim: dimensions of the image

    * at the moment, only using the first camera

    return:
        seq_data_dict: a dictionary where key is sequence number of value is data/target of that sequence
    """

    seq_data_dict = {}

    # iterate over data sequences
    seq_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for seq_dir in seq_dirs:
        n_seq = int(seq_dir[-1])
        seq_data_dict[n_seq] = {'data': None, 'target': None}

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
        seq_data_dict[n_seq]['data'] = seq_data
        seq_data_dict[n_seq]['target'] = seq_target

    return seq_data_dict


seq_data_dict = load_data('assembly_data')

# evaluation
if not os.path.isdir(eval_dir):
    mkdir_p(eval_dir)

print('Begin evaluation...')
model.eval()
lossfun = nn.MSELoss(reduction='none')
for n_seq in seq_data_dict.keys():
    print('Evaluating sequence {}'.format(n_seq))

    seq_dir = os.path.join(eval_dir, '{}'.format(n_seq))
    if not os.path.isdir(seq_dir):
        mkdir_p(seq_dir)

    seq_data = seq_data_dict[n_seq]['data']
    seq_target = seq_data_dict[n_seq]['target']
    seq_data_tensor = torch.FloatTensor(seq_data)
    seq_target_tensor = torch.FloatTensor(seq_target)
    seq_data_tensor = seq_data_tensor.permute(0, 3, 1, 2)
    seq_target_tensor = seq_target_tensor.permute(0, 3, 1, 2)
    seq_data_var, seq_target_var = Variable(seq_data_tensor), Variable(seq_target_tensor)
    seq_output_tensor = model(seq_data_var)
    seq_output = seq_output_tensor.permute(0, 2, 3, 1).detach().numpy()

    prediction_error = lossfun(seq_output_tensor, seq_target_var)
    prediction_error = np.mean(prediction_error.detach().numpy(), axis=(1, 2, 3))
    timesteps = range(len(prediction_error))

    # plot prediction error vs timesteps
    plot_path = os.path.join(seq_dir, 'pred_error.png')
    plt.figure(num=1, figsize=(8, 6))
    plt.plot(timesteps, prediction_error)
    plt.xlabel("Timesteps", fontsize=20)
    plt.ylabel("Prediction Error", fontsize=20)
    plt.title("Sequence {} Prediction Error Plot".format(n_seq), fontsize=20)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # lowest
    lowest_error_idx = np.argpartition(prediction_error, num_key_frames)[:num_key_frames]
    lowest_dir = os.path.join(seq_dir, 'lowest')
    if not os.path.isdir(lowest_dir):
        mkdir_p(lowest_dir)

    for idx in lowest_error_idx:
        idx_dir = os.path.join(lowest_dir, '{}'.format(idx))
        if not os.path.isdir(idx_dir):
            mkdir_p(idx_dir)
        data = seq_data[idx]
        target = seq_target[idx]
        output = seq_output[idx]

        # sliding window
        data_path = os.path.join(idx_dir, 'sliding_window_{}.png'.format(idx))
        fig, axes = plt.subplots(1, window_size)
        fig.suptitle('sliding_window_{}'.format(idx), fontsize=20)
        for axis_num in range(len(axes)):
            ax = axes[axis_num]
            ax.imshow(data[:, :, (3 * axis_num):(3 * (axis_num + 1))])
            ax.set_title('Past frame {}'.format(axis_num + 1))
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(data_path)
        plt.close(fig)

        # target
        target_path = os.path.join(idx_dir, 'target_{}.png'.format(idx))
        target_img = Image.fromarray((target * 255).astype(np.uint8))
        target_img.save(target_path)

        # output
        output_path = os.path.join(idx_dir, 'output_{}.png'.format(idx))
        output_img = Image.fromarray((output * 255).astype(np.uint8))
        output_img.save(output_path)

    # highest
    highest_error_idx = np.argpartition(prediction_error, -num_key_frames)[-num_key_frames:]
    highest_dir = os.path.join(seq_dir, 'highest')
    if not os.path.isdir(highest_dir):
        mkdir_p(highest_dir)

    for idx in highest_error_idx:
        idx_dir = os.path.join(highest_dir, '{}'.format(idx))
        if not os.path.isdir(idx_dir):
            mkdir_p(idx_dir)
        data = seq_data[idx]
        target = seq_target[idx]
        output = seq_output[idx]

        # sliding window
        data_path = os.path.join(idx_dir, 'sliding_window_{}.png'.format(idx))
        fig, axes = plt.subplots(1, window_size)
        fig.suptitle('sliding_window_{}'.format(idx), fontsize=20)
        for axis_num in range(len(axes)):
            ax = axes[axis_num]
            ax.imshow(data[:, :, (3 * axis_num):(3 * (axis_num + 1))])
            ax.set_title('Past frame {}'.format(axis_num + 1))
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(data_path)
        plt.close(fig)

        # target
        target_path = os.path.join(idx_dir, 'target_{}.png'.format(idx))
        target_img = Image.fromarray((target * 255).astype(np.uint8))
        target_img.save(target_path)

        # output
        output_path = os.path.join(idx_dir, 'output_{}.png'.format(idx))
        output_img = Image.fromarray((output * 255).astype(np.uint8))
        output_img.save(output_path)






