from torchvision import datasets, transforms
from utils.sampling import iid, noniid, iid_unbalanced, noniid_unbalanced
import torch
from torch.utils.data import ConcatDataset
import numpy as np
import random
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
# from utils.femnist import FEMNIST
from utils.leaf import CusteomLEAF

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.augment_utils import Mod_PrefetchGenerator, Mod_DataGenerator
import os

trans_mnist = transforms.Compose([
    transforms.ToTensor(), # TODO: channel is 1
    transforms.Normalize((0.1307,), (0.3081,)),
])
trans_emnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
trans_celeba = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

trans_cifar100_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])
trans_cifar100_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])
trans_cifar100_pretrain = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])

trans_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trans_cifar10_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trans_cifar10_pretrain = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def _split_according_to_prior(label, client_num, prior):  
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))  
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)  

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]  
        np.random.shuffle(idx_k)
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] * len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):  
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, np.cumsum(nums_k)[:-1]))]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def dirichlet_distribution_noniid_slice(label, client_num, alpha, min_size=1, prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    partition/noniid_partition.py
    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError("Only support single-label tasks!")

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f"The number of sample should be " f"greater than" f" {client_num * min_size}."
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            # prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            # ])
            # prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            if client_num <= 400:
                idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))]
            else:
                idx_k_slice = [idx.tolist() for idx in np.split(idx_k, prop)]
                idxs = np.arange(len(idx_k_slice))
                np.random.shuffle(idxs)
                idx_slice = [idx_j + idx_k_slice[idx] for idx_j, idx in zip(idx_slice, idxs)]

            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])

    dict_users = {client_idx: np.array(idx_slice[client_idx]) for client_idx in range(client_num)}
    return dict_users # idx_slice

# <<< data split from pFL-Bench

def get_data(args, env='fed'):

    total_users = args.num_users + args.ood_users

    if env == 'single':
        if args.dataset == 'cifar10':
            dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
            dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)

        elif args.dataset == 'cifar100':
            dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        return dataset_train, dataset_test

    elif env == 'fed':
        if args.unbalanced:
            if args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, total_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, total_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, total_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, total_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, total_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, total_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, total_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, total_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        else:
            if args.dataset == 'mnist':
                dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
                dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
                # sample users
                if args.iid:
                    dict_users_train = iid(dataset_train, total_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, total_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, total_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, total_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, total_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, total_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, total_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, total_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, total_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, total_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, total_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, total_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        return dataset_train, dataset_test, dict_users_train, dict_users_test

    elif env == 'pfl-bench':
        if not args.unbalanced:
            # handle multiple datasets
            if len(args.dataset.split(',')) > 1:
                dataset_list = args.dataset.split(',')
                dataset_train_list, dataset_test_list, dict_users_train_list, dict_users_test_list = [], [], [], []
                classes = []
                for dataset in dataset_list:
                    args.dataset = dataset
                    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args, env='pfl-bench')
                    dataset_train.targets = torch.tensor(dataset_train.targets) + len(classes)
                    dataset_test.targets = torch.tensor(dataset_test.targets) + len(classes)
                    dataset_train_list.append(dataset_train)
                    dataset_test_list.append(dataset_test)
                    dict_users_train_list.append(dict_users_train)
                    dict_users_test_list.append(dict_users_test)
                    classes += dataset_train.classes
                args.dataset = ",".join(dataset_list)
                dataset_train = ConcatDataset(dataset_train_list)
                dataset_train.classes = classes
                dataset_train.targets = torch.concat([dataset_train.targets for dataset_train in dataset_train_list])
                dataset_test = ConcatDataset(dataset_test_list)
                dataset_test.classes = classes
                dataset_test.targets = torch.concat([dataset_test.targets for dataset_test in dataset_test_list])
                dict_users_train = dict()
                dict_users_test = dict()
                for i in range(total_users):
                    dict_users_train[i] = np.array([], dtype=np.int64)
                    dict_users_test[i] = np.array([], dtype=np.int64)
                    for j in range(len(dataset_list)):
                        dict_users_train[i] = np.concatenate((dict_users_train[i], dict_users_train_list[j][i]))
                        dict_users_test[i] = np.concatenate((dict_users_test[i], dict_users_test_list[j][i]))
                print(f'Multiple dataset {args.dataset} loaded')
                print(f'users          : {len(dict_users_train)}')
                print(f'train data size: {len(dataset_train)}')
                print(f'test data size : {len(dataset_test)}')
                print(f'classes ({len(classes)})   : {classes}')
                all_targets = set()
                for dataset in dataset_train.datasets:
                    all_targets.update(set(dataset.targets.numpy()))
                assert len(classes) == len(all_targets)
                return dataset_train, dataset_test, dict_users_train, dict_users_test

            # handle single dataset
            if args.dataset == 'femnist':
                leaf_dataset = CusteomLEAF('data', 'femnist', s_frac=getattr(args, 's_frac', 1.0 if total_users > 400 else 0.1), tr_frac=getattr(args, 'tr_frac', 0.8), val_frac=getattr(args, 'val_frac', 0.0), seed=args.seed, transform=trans_emnist)
                dataset_train = leaf_dataset.dataset_train
                dataset_test = leaf_dataset.dataset_test
                dict_users_train = leaf_dataset.dict_users_train
                dict_users_test = leaf_dataset.dict_users_test
                print(f'LEAF dataset {args.dataset} loaded')
                print(f'users          : {len(dict_users_train)}')
                print(f'train data size: {len(dataset_train)}')
                print(f'test  data size: {len(dataset_test)}')
                print(f'total data size: {sum([len(dict_users_train[i]) for i in dict_users_train])}')
                return dataset_train, dataset_test, dict_users_train, dict_users_test

            elif args.dataset == 'celeba':
                leaf_dataset = CusteomLEAF('data', 'celeba', s_frac=getattr(args, 's_frac', 1.0 if total_users > 500 else 0.05), tr_frac=getattr(args, 'tr_frac', 0.8), val_frac=getattr(args, 'val_frac', 0.0), seed=args.seed, transform=trans_celeba)
                dataset_train = leaf_dataset.dataset_train
                dataset_test = leaf_dataset.dataset_test
                dict_users_train = leaf_dataset.dict_users_train
                dict_users_test = leaf_dataset.dict_users_test
                print(f'LEAF dataset {args.dataset} loaded')
                print(f'users          : {len(dict_users_train)}')
                print(f'train data size: {len(dataset_train)}')
                print(f'test  data size: {len(dataset_test)}')
                print(f'total data size: {sum([len(dict_users_train[i]) for i in dict_users_train])}')
                return dataset_train, dataset_test, dict_users_train, dict_users_test

            elif args.dataset == 'mnist':
                dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
                dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
            elif args.dataset == 'cifar10':
                if args.augment_type == 1: # Pretrain (default federated learning) 
                    dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                    dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                elif args.augment_type == 2: # Search (Normalize only for policy search) 
                    dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=None)
                    dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
            elif args.dataset == 'cifar100':
                if args.augment_type == 1: # Pretrain (default federated learning) 
                    dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                    dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                elif args.augment_type == 2: # Search (Normalize only for policy search) 
                    dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=None)
                    dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
            elif args.dataset == 'emnist':
                dataset_train = datasets.EMNIST('data/emnist', 'byclass', train=True, download=True, transform=trans_emnist)
                dataset_test = datasets.EMNIST('data/emnist', 'byclass', train=False, download=True, transform=trans_emnist)
            else:
                exit('Error: unrecognized dataset')

            if args.iid:
                dict_users_train = iid(dataset_train, total_users, args.server_data_ratio)
                dict_users_test = iid(dataset_test, total_users, args.server_data_ratio)
            else:
                dict_users_train = dirichlet_distribution_noniid_slice(np.array(dataset_train.targets), total_users, args.alpha)
                train_label_distribution = [[dataset_train.targets[idx] for idx in dict_users_train[user_idx]] for user_idx in range(total_users)]
                dict_users_test = dirichlet_distribution_noniid_slice(np.array(dataset_test.targets), total_users, args.alpha, prior=train_label_distribution)

        return dataset_train, dataset_test, dict_users_train, dict_users_test



def find_folders(root_dir, lastname):
    matching_folders = []
    for root, dirs, files in os.walk(root_dir):
        for folder in dirs:
            if folder.endswith(lastname):
                matching_folders.append(os.path.join(root, folder))
    assert len(matching_folders) == 1
    return matching_folders[0]

from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def clip_norm_(grads, max_norm, norm_type: float = 2.0):
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    clip_coef = max_norm / (total_norm  + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))

    return total_norm

def clip_norm_coef(grads, max_norm, norm_type: float = 2.0):
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in grads]), norm_type)
    clip_coef = max_norm / (total_norm  + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    return clip_coef_clamped.to(device)


def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels_oneh):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE


def draw_reliability_graph(preds, labels_oneh):
#   import ipdb; ipdb.set_trace(context=5)
  ECE, MCE = get_metrics(preds, labels_oneh)
  bins, _, bin_accs, _, _ = calc_bins(preds, labels_oneh)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True)
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  #plt.savefig('calibrated_network.png', bbox_inches='tight')
  #plt.close(fig)
  return fig, ECE, MCE


def plot_data_partition(dataset, dict_users, num_classes, num_sample_users, writer=None, tag="Data Partition"):
    dict_users_targets={}
    targets=np.array(dataset.targets)

    dict_users_targets = {client_idx: targets[data_idxs] for client_idx, data_idxs in dict_users.items()}

    s=torch.stack([torch.bincount(torch.tensor(data_idxs), minlength=num_classes) for client_idx, data_idxs in dict_users_targets.items()])
    ss=torch.cumsum(s, 1)
    cmap = plt.cm.get_cmap('hsv', num_classes)
    fig, ax = plt.subplots(figsize=(20, num_sample_users))
    ax.barh([f"Client {i:3d}" for i in range(num_sample_users)], s[:num_sample_users, 0], color=cmap(0))
    for c in range(1, num_classes):
        ax.barh([f"Client {i:3d}" for i in range(num_sample_users)], s[:num_sample_users, c], left=ss[:num_sample_users, c-1], color=cmap(c))
    # plt.show()
    if writer is not None:
        writer.add_figure(tag, fig)

def repeat(x, n, axis):
    if isinstance(x, np.ndarray):
        return np.repeat(x, n, axis=axis)
    elif isinstance(x, list):
        return repeat_list(x, n, axis)
    else:
        raise Exception('Unsupport data type {}'.format(type(x)))

def repeat_list(x, n, axis):
    assert isinstance(x, list), 'Can only consume list type'
    if axis == 0:
        x_new = sum([[x_] * n for x_ in x], [])
    elif axis > 1:
        x_new = [repeat(x_, n, axis=axis - 1) for x_ in x]
    else:
        raise Exception
    return x_new 


def plot_metaaug_policyhp(ml, idx, hist_info, args, vmin=None, vmax=None, cmap="viridis", policy=None):
    # misc
    ops_dense, mags_dense, reduce_random_mat, ops_mags_idx = ml.policynet.get_dense_aug(None,False)
    op_names = ml.policynet.op_names
    n_stage = len(ml.policynet.server_policy)
    n_op = len(ops_dense)
    if policy is None:
        init_prob = ml.adapted_policy[idx][0].clone().detach()
        init_prob_np = init_prob.reshape(n_op,n_op).numpy()
    else:
        init_prob_np = policy

    op_labels = [name.split(':')[0] for name in op_names]
    # Creating figure and axes objects
    fig, ax = plt.subplots(figsize=(12, 10))
    # Plotting
    cax = ax.imshow(init_prob_np, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    # Setting labels and title
    fontdict = {"fontsize": 16}
    ax.set_xticks(np.arange(len(op_labels)))
    ax.set_yticks(np.arange(len(op_labels)))
    ax.set_xticklabels(op_labels, fontdict=fontdict, rotation=90)
    ax.set_yticklabels(op_labels, fontdict=fontdict)
    ax.xaxis.set_ticks_position("top")
    fig.colorbar(cax, ax=ax, label=None)
    fig.tight_layout()
    return fig 


def plot_metaaug_policyhp_ind_mag(ml, idx, hist_info, args):

    # misc
    ops_dense, mags_dense, reduce_random_mat, ops_mags_idx = ml.policynet.get_dense_aug(None,False)
    op_names = ml.policynet.op_names
    n_stage = len(ml.policynet.server_policy)
    n_op = len(ops_dense)

    # Converting torch tensor to numpy array for visualization
    init_prob = ml.adapted_policy[idx][0].clone().detach() # torch.Size([324])
    init_prob_np = init_prob.reshape(n_op,n_op).numpy()
    op_labels = [name.split(':')[0] for name in op_names]

    # Creating figure and axes objects
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plotting the heatmap
    cax = ax[0].imshow(init_prob_np, cmap='viridis', interpolation='nearest')

    # # Creating figure and axes objects
    # fig, ax = plt.subplots(figsize=(12, 10))
    # # Plotting
    # cax = ax.imshow(init_prob_np, cmap='viridis', interpolation='nearest')

    # Setting labels and title for the heatmap
    ax[0].set_xticks(np.arange(len(op_labels)))
    ax[0].set_yticks(np.arange(len(op_labels)))
    ax[0].set_xticklabels(op_labels, rotation=90)
    ax[0].set_yticklabels(op_labels)
    fig.colorbar(cax, ax=ax[0], label='Value')
    ax[0].set_title('Operations Heatmap')
    ax[0].set_xlabel('Operation Name')
    ax[0].set_ylabel('Operation Name')

    # Assuming mag is a torch tensor, converting it to a numpy array
    mag = ml.adapted_mag[idx][0].clone().detach() # torch.Size([10])
    mag_np = mag.numpy()

    # Plotting the mag values
    ax[1].bar(np.arange(len(mag_np)), mag_np)

    # Setting labels and title for the mag plot
    ax[1].set_title('Mag Values')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Value')

    # Adjusting layout to prevent overlap
    plt.tight_layout()

    return fig


class datataset_prepare(Dataset):
    def __init__(self, dataset, client_idx, args):
        np.random.shuffle(client_idx)
        self.data_x = dataset.data[client_idx]
        self.data_y = np.array(dataset.targets)[client_idx]
        self.val_bs = args.val_bs
        self.all_indices = set(range(len(self.data_y)))   
        self.eval = not (len(client_idx) > (args.val_bs + args.local_bs)) 

    def __getval__(self, idx):
        exclude_indices = set(idx.numpy())   
        remaining_indices = list(self.all_indices - exclude_indices)   
        
        batch_indices = random.sample(remaining_indices, self.val_bs)
        val_x = [self.data_x[k] for k in batch_indices]
        val_y = np.array([self.data_y[k] for k in batch_indices])

        val_x = torch.tensor(np.array(val_x))
        val_y = torch.tensor(val_y, dtype=torch.long)

        return val_x, val_y
    
    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        train_x = self.data_x[idx] 
        train_y = np.array(self.data_y[idx]) 
        return train_x, train_y, idx


def get_hist(dataset, client_idx, num_classes):
    hist_info = torch.histc(torch.Tensor(dataset.targets)[client_idx], bins=num_classes, min=0, max=num_classes-1)
    hist_info = hist_info/sum(hist_info)

    return hist_info
