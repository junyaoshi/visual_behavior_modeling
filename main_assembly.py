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
import pickle
import glob
import shutil
import itertools
import numpy as np
from tqdm import tqdm
from utils import Bar, Logger, AverageMeter, regression_accuracy, mkdir_p, savefig, load_data, assembly_dataset, \
    gradient_loss
from new_models.assembly_model import TrajPredictor


# ------------------------------------------- training settings ------------------------------------------- #
TRAIN_BATCH = 128
TEST_BATCH = 100
LR = 0.01
SCHEDULE = [10, 30, 50, 80]
GAMMA = 0.5
EPOCHS = 50
SEED = 1
WORKERS = 4
DATA_DIR = 'data_11_30'
CHECKPOINT = 'checkpoint'
TRAIN_RATIO = 0.8
SAVE_TEST_RESULTS = False
N_CLASSES = 3
WINDOW_SIZE = 2
USE_GRADIENT_LOSS = True


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def addDateTime(s = ""):
    """
    Adds the current date and time at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    import datetime
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16] + date[17:19]
    return s + '_D' + date


def train():
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    scores = []
    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):

        data = data.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)


        data_time.update(time.time() - end)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = lossfun(output, target)
        if args.use_gradient_loss:
            loss = loss + gradient_loss(output, target)

        # measure accuracy and record loss
        prec1 = 0.0
        losses.update(loss.item(), data.size(0))
        top1.update(prec1, data.size(0))

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top1.avg,
                    )
        bar.next()

    return (losses.avg, -top1.avg)


def test(save_flag, epoch):
    global best_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    if save_flag:
        saved_test_results = []

    for batch_idx, (data, target) in enumerate(tqdm(testloader)):

        data = data.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = lossfun(output, target)
        if args.use_gradient_loss:
            loss = loss + gradient_loss(output, target)

        prec1 = 0.0
        losses.update(loss.item(), data.size(0))
        top1.update(prec1, data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top1.avg,
                    )
        bar.next()

        # save outputs
        if save_flag:
            output = output.cpu().detach().numpy() # (100, 12288)
            target = target.cpu().numpy()
            data = data.cpu().numpy()
            saved_test_results.append([data, target, output])

    if save_flag:
        saved_test_results = np.array(saved_test_results)
        filename = "test_resutls_" + str(epoch) + ".npy"
        filepath = os.path.join(test_results_dir, filename)
        np.save(filepath, saved_test_results)


    bar.finish()

    return losses.avg, -top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Observer Network')
    parser.add_argument('--train-batch', default=TRAIN_BATCH, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=TEST_BATCH, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', type=float, default=LR, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--schedule', type=int, nargs='+', default=SCHEDULE,
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=SEED, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-j', '--workers', default=WORKERS, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-c', '--checkpoint', default=CHECKPOINT, type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-d', '--data-dir', default=DATA_DIR, type=str, metavar='PATH',
                        help='path to data')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO,
                        help='percent of data used for training in train-test split')
    parser.add_argument('--save-test-results', action='store_true', default=SAVE_TEST_RESULTS,
                        help='whether to use gradient loss')
    parser.add_argument('--n-classes', default=N_CLASSES, type=int, metavar='N',
                        help='number of classes')
    parser.add_argument('--window-size', default=WINDOW_SIZE, type=int, metavar='N',
                        help='sliding window size')
    parser.add_argument('--use-gradient-loss', action='store_true', default=USE_GRADIENT_LOSS,
                        help='whether to use gradient loss')

    args = parser.parse_args()

    # torch settings
    state = {k: v for k, v in args._get_kwargs()}
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    best_loss = float("inf")

    # make necessary directories
    results_base_dirname = 'results'
    if not os.path.exists(results_base_dirname):
        os.mkdir(results_base_dirname)

    results_dir = os.path.join(results_base_dirname, 'FFP_{}'.format(args.data_dir))
    results_dir = addDateTime(results_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    checkpoint_dir = os.path.join(results_dir, args.checkpoint)
    if not os.path.isdir(checkpoint_dir):
        mkdir_p(checkpoint_dir)

    if args.save_test_results:
        test_results_dir = os.path.join(results_dir, 'test_results')
        mkdir(test_results_dir)

    # Traning related setup
    model = TrajPredictor(args.window_size * 3)
    model.double()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lossfun = nn.MSELoss()
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='observer')
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # load data
    print('Begin loading data from {}...'.format(args.data_dir))

    # train_data, test_data, train_target, test_target, train_seq_inds, test_seq_inds = load_data(args.data_dir)
    #
    # train_data = torch.FloatTensor(train_data)
    # train_target = torch.FloatTensor(train_target)
    # test_data = torch.FloatTensor(test_data)
    # test_target = torch.FloatTensor(test_target)
    #
    # final_train_data = torch.utils.data.TensorDataset(train_data, train_target)
    # final_test_data = torch.utils.data.TensorDataset(test_data, test_target)

    # train-test split
    seq_dirs = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]
    num_seq = len(seq_dirs)
    split_idx = int(round(args.train_ratio * num_seq))
    np.random.shuffle(seq_dirs)
    train_seqs = seq_dirs[:split_idx]
    test_seqs = seq_dirs[split_idx:]
    print('Split data into {} training sequences and {} testing sequences.'.format(len(train_seqs), len(test_seqs)))

    final_train_data = assembly_dataset(train_seqs, args.window_size)
    final_test_data = assembly_dataset(test_seqs, args.window_size)

    trainloader = torch.utils.data.DataLoader(final_train_data, batch_size=args.train_batch, shuffle=True,
                                              num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(final_test_data, batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers)

    # saving training and testing sequence indices
    with open(os.path.join(results_dir, 'train_seqs.p'), 'wb') as f:
        pickle.dump(train_seqs, f)

    with open(os.path.join(results_dir, 'test_seqs.p'), 'wb') as f:
        pickle.dump(test_seqs, f)

    # np.save(os.path.join(results_dir, 'train_seq_inds.npy'), np.array(train_seq_inds))
    # np.save(os.path.join(results_dir, 'test_seq_inds.npy'), np.array(test_seq_inds))

    # Train
    print('Begin training...')
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        if args.save_test_results:
            if epoch % 10 == 0:
                save_flag = True
            else:
                save_flag = False
        else:
            save_flag = False

        train_loss, train_acc = train()
        test_loss, test_acc = test(save_flag, epoch)
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': test_loss,
            'acc': test_acc,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    print('Training done. Best test loss: ', best_loss)
