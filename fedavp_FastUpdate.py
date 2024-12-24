#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import multiprocessing
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# from tensorboardX import SummaryWriter
from tqdm import trange

from utils.train_utils import get_data, set_seed, find_folders, DatasetSplit, draw_reliability_graph
from utils.train_utils import plot_metaaug_policyhp, datataset_prepare, get_hist

from models.Faal_step import Faal_step
from models.Faal_step_fast import Faal_step_fast

import os

# AA
from utils.augment_utils import get_augmentation, get_lops_luniq, get_mid_magnitude
from utils.train_utils import DatasetSplit, datataset_prepare, draw_reliability_graph, find_folders, get_data, get_hist, plot_metaaug_policyhp, set_seed


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=777)

    # model
    parser.add_argument("--algorithm", type=str, default="faal_step_fast")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=200)

    # federated arguments
    parser.add_argument("--num_rounds", type=int, default=1000, help="rounds of training")  # 500
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument("--ood_users", type=int, default=30)
    parser.add_argument("--frac_m", type=int, default=10, help="the fraction of clients: C")


    parser.add_argument("--local_bs", type=int, default=64, help="local batch size: B")
    parser.add_argument("--local_epochs", type=int, default=5, help="the number of local SGD epochs in local epoch")

    parser.add_argument("--adaptation_bs", type=int, default=64, help="local batch size: B")
    parser.add_argument("--adaptation_epochs", type=int, default=1)
    parser.add_argument("--adaptation_steps", type=int, default=1, help="policy adaptation steps")

    # pfl-bench
    parser.add_argument("--alpha", type=float, default=0.1)

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar100", help="name of dataset")  ### DATA
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")  ### DATA
    parser.add_argument("--unbalanced", action="store_true", help="unbalanced data size")
    parser.add_argument("--num_batch_users", type=int, default=0, help="when unbalanced dataset setting, batch users (same data size)")
    parser.add_argument("--moved_data_size", type=int, default=0, help="when unbalanced dataset setting, moved data size")
    parser.add_argument("--server_data_ratio", type=float, default=0.0, help="The percentage of data that servers also have across data of all clients.")
    parser.add_argument("--shard_per_user", type=int, default=10, help="classes per user")

    parser.add_argument("--pre", type=str, default="")  # ! added

    parser.add_argument(
        "--policy_type",
        type=int,
        default=0, 
    )

    parser.add_argument(
        "--augment_type",
        type=int,
        default=2,
        help="1: Pretrain (Default Federated Learning), \
              2: Search (Normalize only for policy search)", 
    )
    parser.add_argument("--l_mags", type=int, default=0, help="fixed")
    parser.add_argument("--n_cpu", type=int, default=8, help="fixed")

    parser.add_argument('--expansion', type=int, default=1, help='expansion for augment sampling') 
    parser.add_argument('--w_inner_lr', type=float, default=0.2, help="parameter for client model inner lr")
    parser.add_argument('--p_inner_lr', type=float, default=0.4, help="parameter for client policy inner lr")
    parser.add_argument('--H1', type=int, default=25, help='embedding dim H1') 
    parser.add_argument('--H2', type=int, default=25, help='hidden dim H2') 
    parser.add_argument('--gradclip_policy', type=float, default=0.45, help="gradclip of policy for magnitude noise, 0.1~1.0")
    parser.add_argument('--eps', type=float, default=0.1, help="sampling regularization")
    parser.add_argument('--first_order', type=int, default=1, help="first order policy gradient")
    parser.add_argument('--score_norm', type=int, default=0, help="score normalization")

    parser.add_argument("--inner_steps", type=int, default=1, help="parameter for policy adaptation inner step")
    parser.add_argument("--val_bs", type=int, default=64, help="validation data size")
    parser.add_argument("--pwd", type=float, default=0, help="policy weight decay")
    parser.add_argument("--policy_lr", type=float, default=0.7, help="Server policy learning rate")

    args = parser.parse_args()
    return args


def main(args, trial=None):
    set_seed(args.seed)

    ## Get Data
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args, env="pfl-bench")

    ## Divide policy range(continuous) into discrete space, dtype=dict
    ####################################################################################################
    ops_mid_magnitude = get_mid_magnitude(args.l_mags)
    args.l_ops, args.l_uniq = get_lops_luniq(args, ops_mid_magnitude)
    augmentation_default, augmentation_search, augmentation_test = get_augmentation(args)
    pool = multiprocessing.Pool(processes=args.n_cpu)
    ####################################################################################################

    users_datasize = [len(user_len) for user_len in dict_users_train.values()]
    users_pdatasize = [(len(user_len) if len(user_len)>=(args.val_bs+args.local_bs) else 0) for user_len in dict_users_train.values()] ## modified
    num_classes = len(dataset_train.classes)

    

    ml_algorithms = {
        'Faal_step' : Faal_step,
        'Faal_step_fast' : Faal_step_fast,
    }

    ml = ml_algorithms[args.algorithm.capitalize()](args, num_classes)
    assert args.augment_type == 2

    val_loss_list = []
    val_ploss_list = []
    val_acc_list = []
    val_oodacc_list = []
    aug_acc_list = []
    hist_infos = dict()

    adapted_policy_dict = {}


    for idx in range(args.num_users):
        client_idx = dict_users_train[idx]
        hist_info = get_hist(dataset_train, client_idx, num_classes)
        if idx not in hist_infos:
            hist_infos[idx] = hist_info

    step_iter = trange(ml.args.num_rounds)
    for iter in step_iter:
        # Server choose clients to train
        user_idxs = np.random.choice(range(ml.args.num_users), ml.args.frac_m, replace=False)

        collected_policies = []
        collected_weights = []

        val_loss_list.clear()
        val_ploss_list.clear()
        val_acc_list.clear()
        aug_acc_list.clear()

        for idx in user_idxs:
            ################################## Client #####################################
            client_idx = dict_users_train[idx]

            ldr_train = DataLoader(datataset_prepare(dataset_train, client_idx, args), batch_size=args.local_bs, shuffle=True, num_workers=0)

            # Copy weight from SERVER to Client
            wt = ml.hpnet(idx)  # Weight Sampling from posteiror wt ~ q(w*)
            w_local = [w.detach().clone() for w in wt]
            pt = ml.hpnet_policynet[0](idx)
            p_local = [w.detach().clone() for w in pt]

            val_loss = 0
            val_ploss = 0
            val_acc = 0
            aug_acc = 0

            for i in range(ml.args.local_epochs):
                wt, p_local, mean_loss, mean_ploss, mean_valacc, mean_augacc, weights_2_dict = ml.client_update(ldr_train, wt, p_local, augmentation_search, augmentation_test, pool, args, 0, hist_info, idx, w_local, round=iter, local_epoch=i)
                val_loss += mean_loss
                val_ploss += mean_ploss
                val_acc += mean_valacc
                aug_acc += mean_augacc

                if iter%10 == 0 or iter == (ml.args.num_rounds-1):
                    adapted_policy_dict.update(weights_2_dict)
             
            val_loss /= ml.args.local_epochs
            val_ploss /= ml.args.local_epochs
            val_acc /= ml.args.local_epochs
            aug_acc /= ml.args.local_epochs

            val_loss_list.append(val_loss.item())
            val_ploss_list.append(val_ploss.item())
            val_acc_list.append(val_acc)
            aug_acc_list.append(aug_acc)

            ############################################################################
            collected_policies.append( p_local )
            collected_weights.append( [w.detach().clone() for w in wt] )

        ############################### Server ########################################
        ml.server_policy_aggregation(user_idxs, users_pdatasize, [collected_policies]) ## modified
        ml.server_aggregation(user_idxs, users_datasize, [collected_weights]) 
        step_iter.set_description(f"Step:{iter}, AVG Loss: {np.array(val_loss_list).mean():.4f},  VAL AVG Acc: {np.array(val_acc_list).mean():.4f}, AUG AVG Acc: {np.array(aug_acc_list).mean():.4f}")


        giter = 0


        if iter % 100 == 0 or iter == (ml.args.num_rounds - 1):  # Testing
            val_loss_list.clear()
            val_acc_list.clear()
            aug_acc_list.clear()

            preds_list = []
            labels_oneh_list = []

            if iter == (ml.args.num_rounds - 1):
                user_idxs = range(ml.args.num_users)
                set_seed(args.seed)  # ! data loader seed setting
            else:
                num_test_users = 50
                user_idxs = np.random.choice(range(ml.args.num_users), num_test_users, replace=False)
                random_hist = [0, 20, 40, 60, 80]
                for value in random_hist:
                    if value not in user_idxs:
                        user_idxs = np.append(user_idxs, value)
            #
             
            for idx in user_idxs:
                wt = ml.hpnet(idx)  # Weight Sampling from posteiror wt ~ q(w*)
                w_local = [w.detach().clone() for w in wt]
                pt = ml.hpnet_policynet[0](idx)
                p_local = [w.detach().clone() for w in pt]
                client_idx_train = dict_users_train[idx]

                ldr_train = DataLoader(datataset_prepare(dataset_train, client_idx_train, args), batch_size=args.local_bs, shuffle=True, num_workers=0)
                ldr_test = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size=100, shuffle=True)

                for i in range(args.adaptation_epochs):
                    wt = ml.client_adapt(ldr_train, wt, p_local, augmentation_search, augmentation_test, pool, args, 0, hist_info, idx, w_local)

                if iter == (ml.args.num_rounds - 1):  # Measure Calibration
                    mean_loss, mean_acc, preds, labels_oneh = ml.client_test_with_calibration(ldr_test, wt)
                    preds_list.append(preds)
                    labels_oneh_list.append(labels_oneh)
                else:
                    mean_loss, mean_acc = ml.client_test(ldr_test, wt)

                val_loss_list.append(mean_loss)
                val_acc_list.append(mean_acc)



            print(f"Step:{iter}, (test) AVG Loss: {np.array(val_loss_list).mean():.4f}, (test) AVG Acc: {np.array(val_acc_list).mean():.4f}")
            #  uniform average
            ml.writer.add_scalar("test/AVG_loss", np.array(val_loss_list).mean(), iter)
            ml.writer.add_scalar("test/AVG_acc", np.array(val_acc_list).mean() * 100, iter)
            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights = torch.Tensor(weights_size) / sum(weights_size)
            ml.writer.add_scalar("test/weighted_loss", np.average(val_loss_list, weights=weights), iter)
            ml.writer.add_scalar("test/weighted_acc", np.average(val_acc_list, weights=weights) * 100, iter)
            ml.writer.flush()


        if iter % 100 == 0 or iter == (ml.args.num_rounds - 1):  # OOD
            val_loss_list.clear()
            val_oodacc_list.clear()

            preds_list = []
            labels_oneh_list = []

            if iter == (ml.args.num_rounds - 1):
                set_seed(args.seed)  # ! data loader seed setting

            user_idxs = range(ml.args.num_users, ml.args.num_users + ml.args.ood_users)
            for idx in user_idxs:
                ldr_train = DataLoader(datataset_prepare(dataset_train, client_idx_train, args), batch_size=args.local_bs, shuffle=True, num_workers=0)
                ldr_test = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size=100, shuffle=True)

                wt = ml.hpnet(idx)  # Weight Sampling from posteiror wt ~ q(w*)
                w_local = [w.detach().clone() for w in wt]
                pt = ml.hpnet_policynet[0](idx)
                p_local = [w.detach().clone() for w in pt]

                for i in range(args.adaptation_epochs):
                    wt = ml.client_adapt(ldr_train, wt, p_local, augmentation_search, augmentation_test, pool, args, 0, hist_info, idx, w_local)

                if iter == (ml.args.num_rounds - 1):  # Measure Calibration
                    mean_loss, mean_acc, preds, labels_oneh = ml.client_test_with_calibration(ldr_test, wt)
                    preds_list.append(preds)
                    labels_oneh_list.append(labels_oneh)
                else:
                    mean_loss, mean_acc = ml.client_test(ldr_test, wt)

                val_loss_list.append(mean_loss)
                val_oodacc_list.append(mean_acc)



            print(f"Step:{iter}, (OOD) AVG Loss: {np.array(val_loss_list).mean():.4f}, (OOD) AVG Acc: {np.array(val_oodacc_list).mean():.4f}")
            #  uniform average
            ml.writer.add_scalar("OOD/AVG_loss", np.array(val_loss_list).mean(), iter)
            ml.writer.add_scalar("OOD/AVG_acc", np.array(val_oodacc_list).mean() * 100, iter)
            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights2 = torch.Tensor(weights_size) / sum(weights_size)
            ml.writer.add_scalar("OOD/weighted_loss", np.average(val_loss_list, weights=weights2), iter)
            ml.writer.add_scalar("OOD/weighted_acc", np.average(val_oodacc_list, weights=weights2) * 100, iter)
            ml.writer.flush()


    torch.cuda.empty_cache()
    return np.average(val_acc_list, weights=weights) * 100, np.average(val_oodacc_list, weights=weights2) * 100


if __name__ == "__main__":
    args = args_parser()
    main(args, None)
