from __future__ import print_function
import argparse
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import pickle
from tqdm import tqdm
import time
import glob
import shutil
import itertools
import numpy as np
from utils import Bar, Logger, AverageMeter, regression_accuracy, mkdir_p, savefig, assembly_dataset, gradient_loss
from new_models.assembly_model import TrajPredictor

eval_dir = 'assembly_evaluation'
results_dir = os.path.join('results', 'FFP_data_11_30_D201205_023611')
window_size = 2
num_key_frames = 4
sample_seqs = True
sample_train_seqs_num = 5
sample_test_seqs_num = 5
img_dim = (128, 128, 3)
num_workers = 0
model_path = os.path.join('checkpoint', 'model_best.pth.tar')
model = TrajPredictor(window_size * 3)
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()

# load training and testing sequence indices
with open(os.path.join(results_dir, 'train_seqs.p'), 'rb') as f:
    train_seqs = pickle.load(f)

with open(os.path.join(results_dir, 'test_seqs.p'), 'rb') as f:
    test_seqs = pickle.load(f)

if sample_seqs:
    train_seqs = np.random.choice(train_seqs, sample_train_seqs_num)
    test_seqs = np.random.choice(test_seqs, sample_test_seqs_num)

train_dataset = assembly_dataset(train_seqs, window_size)
test_dataset = assembly_dataset(test_seqs, window_size)

# evaluation
if not os.path.isdir(eval_dir):
    mkdir_p(eval_dir)

print('Begin evaluation...')
model.eval()
lossfun = nn.MSELoss(reduction='none')

for mode in ['train', 'teest']:
    if mode == 'train':
        data_dirs = train_seqs
        dataset = train_dataset
    else:
        data_dirs = test_seqs
        dataset = test_dataset
    print('Begin evaluation of {}ing sequences...'.format(mode))

    for data_dir_idx in tqdm(range(len(data_dirs))):
        data_dir = data_dirs[data_dir_idx]
        print('Evaluating sequence {}'.format(data_dir))

        _, n_seq = os.path.split()
        seq_dir = os.path.join(eval_dir, '{}'.format(n_seq))
        if not os.path.isdir(seq_dir):
            mkdir_p(seq_dir)

        data_idx_left = 0 if data_dir_idx == 0 else dataset.data_bins[data_dir_idx - 1]
        data_idx_right = dataset.data_bins[data_dir_idx]
        data_subset = Subset(dataset, range(data_idx_left, data_idx_right))
        dataloader = DataLoader(data_subset, batch_size=data_subset.__len__(),
                                shuffle=False, num_workers=num_workers)

        seq_data, seq_target = next(iter(dataloader))
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






