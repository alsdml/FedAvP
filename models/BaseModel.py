import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from utils.train_utils import get_data, set_seed
import uuid, os
from utils.train_utils import repeat
from utils.train_utils import clip_norm_, clip_norm_coef
import torchvision

def numpy2tensor(batch_input):
    L = batch_input.shape[0]
    output = torch.Tensor(L, 3, 32, 32)
    TT = torchvision.transforms.ToTensor()
    for i in range(L):
        output[i] = TT(batch_input[i])
    return output

class BaseModel:
    def __init__(self, args):

        #### TensorBoard Writer ####
        args.log_dir = f'./runs/{args.dataset}/log' + '/' + args.exp
        args.ckpt_dir = f'./runs/{args.dataset}/ckpt' + '/' + args.exp
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
        self.writer = SummaryWriter(args.log_dir)

        self.criteria = torch.nn.CrossEntropyLoss()
        self.per_criteria = torch.nn.CrossEntropyLoss(reduction='none')

        self.args = args

    def faal_train_report(self,iter,val_loss_list, val_acc_list, aug_acc_list, user_idxs, users_datasize, stage):
            
            giter = int(stage*self.args.num_rounds)

            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights = torch.Tensor(weights_size) / sum(weights_size)
            self.writer.add_scalar('train_stage_all/Weighted_loss', np.average(val_loss_list, weights=weights), iter+giter)
            self.writer.add_scalar('train_stage'+str(stage)+'/Weighted_loss', np.average(val_loss_list, weights=weights), iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/Weighted_acc', np.average(val_acc_list, weights=weights)*100, iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/Weighted_augacc', np.average(aug_acc_list, weights=weights)*100, iter)

            #  uniform average
            self.writer.add_scalar('train_stage_all/AVG_loss', np.array(val_loss_list).mean(), iter+giter)
            self.writer.add_scalar('train_stage'+str(stage)+'/AVG_loss', np.array(val_loss_list).mean(), iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/AVG_acc', np.array(val_acc_list).mean()*100, iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/AVG_augacc', np.array(aug_acc_list).mean()*100, iter)

            self.writer.flush()

    # def faal_train_report_joint(self,iter,val_loss_list, val_ploss_list, val_acc_list, aug_acc_list, user_idxs, users_datasize, stage):
    def faal_train_report_joint(self,iter,val_loss_list, val_ploss_list, val_acc_list, aug_acc_list, user_idxs, users_datasize, users_pdatasize, stage):
            
            giter = int(stage*self.args.num_rounds)

            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights = torch.Tensor(weights_size) / sum(weights_size)

            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_pdatasize[idx])
            pweights = torch.Tensor(weights_size) / sum(weights_size)

            # self.writer.add_scalar('train_stage_all/Weighted_loss', np.average(val_loss_list, weights=weights), iter+giter)
            self.writer.add_scalar('train_stage'+str(stage)+'/Weighted_loss', np.average(val_loss_list, weights=weights), iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/Weighted_ploss', np.average(val_ploss_list, weights=pweights), iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/Weighted_acc', np.average(val_acc_list, weights=pweights)*100, iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/Weighted_augacc', np.average(aug_acc_list, weights=weights)*100, iter)

            #  uniform average
            # self.writer.add_scalar('train_stage_all/AVG_loss', np.array(val_loss_list).mean(), iter+giter)
            self.writer.add_scalar('train_stage'+str(stage)+'/AVG_loss', np.array(val_loss_list).mean(), iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/AVG_ploss', np.array(val_ploss_list).mean(), iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/AVG_acc', np.array(val_acc_list).mean()*100, iter)
            self.writer.add_scalar('train_stage'+str(stage)+'/AVG_augacc', np.array(aug_acc_list).mean()*100, iter)

            self.writer.flush()

    def faal_test_report(self,iter,val_loss_list, val_acc_list, aug_acc_list, user_idxs, users_datasize, stage):
            
            giter = int(stage*self.args.num_rounds)

            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights = torch.Tensor(weights_size) / sum(weights_size)
            self.writer.add_scalar('test_stage_all/Weighted_loss', np.average(val_loss_list, weights=weights), iter+giter)
            self.writer.add_scalar('test_stage'+str(stage)+'/Weighted_loss', np.average(val_loss_list, weights=weights), iter)
            self.writer.add_scalar('test_stage'+str(stage)+'/Weighted_acc', np.average(val_acc_list, weights=weights)*100, iter)
            self.writer.add_scalar('test_stage'+str(stage)+'/Weighted_augacc', np.average(aug_acc_list, weights=weights)*100, iter)

            #  uniform average
            self.writer.add_scalar('test_stage_all/AVG_loss', np.array(val_loss_list).mean(), iter+giter)
            self.writer.add_scalar('test_stage'+str(stage)+'/AVG_loss', np.array(val_loss_list).mean(), iter)
            self.writer.add_scalar('test_stage'+str(stage)+'/AVG_acc', np.array(val_acc_list).mean()*100, iter)
            self.writer.add_scalar('test_stage'+str(stage)+'/AVG_augacc', np.array(aug_acc_list).mean()*100, iter)

            self.writer.flush()


    def train_report(self,iter,val_loss_list, val_acc_list, val_kl_list, wt, user_idxs, users_datasize):
            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights = torch.Tensor(weights_size) / sum(weights_size)
            self.writer.add_scalar('train/weighted_loss', np.average(val_loss_list, weights=weights), iter)
            self.writer.add_scalar('train/weighted_kl', np.average(val_kl_list, weights=weights), iter)
            self.writer.add_scalar('train/weighted_acc', np.average(val_acc_list, weights=weights)*100, iter)

            #  uniform average
            self.writer.add_scalar('train/AVG_loss', np.array(val_loss_list).mean(), iter)
            self.writer.add_scalar('train/AVG_kl', np.array(val_kl_list).mean(), iter)
            self.writer.add_scalar('train/AVG_acc', np.array(val_acc_list).mean()*100, iter)

            self.writer.flush()

    def client_test(self, ldr_test, wt):
        with torch.no_grad():
            val_loss = 0
            val_acc = 0

            
            ## Then we perform validation on the rest of data
            for batch_idx, (images, labels) in enumerate(ldr_test):
                x = images.cuda()
                y = labels.cuda()

                y_pred = self.functional(wt, x.cuda())
                # loss = self.criteria(y_pred, y)
                loss = self.per_criteria(y_pred, y)
                # val_loss += loss.mean().item()
                val_loss += loss.sum().item()
                val_acc += y_pred.argmax(1).eq(y).sum().item()# / len(y)

            # val_loss /= (batch_idx+1)
            # val_acc /= (batch_idx+1)
            val_loss /= len(ldr_test.dataset)
            val_acc /= len(ldr_test.dataset)

        return  val_loss, val_acc
    


    def client_faalhp_test(self, ldr_test, wt, p_local, aug_search, aug_test, pool, args, stage, hist_info):
        loss_list = []
        valacc_list = []
        augacc_list = []

        EXP = 1
        #ops_dense, mags_dense, reduce_random_mat, ops_mags_idx = self.policynet.get_dense_aug(None,False)

        wt_val = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in wt]
        wt_aug = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in wt]

        images_val, labels_val, images_aug, labels_aug = ldr_test.next()
        val_bs = len(labels_val[0])
        images_val, labels_val = aug_test(sum(images_val,[]), np.concatenate(labels_val),
                                              np.array([[0]]*args.search_bs*val_bs, dtype=np.int32),
                                              np.array([[0]]*args.search_bs*val_bs, dtype=np.float32)/float(args.l_mags-1),
                                              use_post_aug=True, pool=pool, chunksize=None)
        images_val = np.reshape(images_val, [val_bs, 32, 32, 3])
        labels_val = np.reshape(labels_val, [val_bs])

        ## sampling augmentation
        EXP_lrn = 100
        images_aug = repeat(images_aug, EXP_lrn, axis=0)
        labels_aug = repeat(labels_aug, EXP_lrn, axis=0)

        ops_k, mags_k = [], []
        for k_stage in range(stage+1):
            dummy_images = [None] * args.search_bs * EXP_lrn
            ops_k_, mags_k_ = self.policynet.sample(dummy_images, dummy_images, None, k_stage, hist_info)
            ops_k.append(ops_k_)
            mags_k.append(mags_k_)
        ops_k = np.concatenate(ops_k, axis=1)
        mags_k = np.concatenate(mags_k, axis=1)
        images_aug, labels_aug = aug_search(sum(images_aug,[]), np.concatenate(labels_aug, axis=0),
                                                ops_k, mags_k.astype(np.float32)/float(args.l_mags-1),
                                                use_post_aug=False, pool=pool, chunksize=None)

        images_aug = np.reshape(images_aug, [EXP_lrn, 32, 32, 3])
        labels_aug = np.reshape(labels_aug, [EXP_lrn])


        x = numpy2tensor(images_val).cuda()
        y = torch.LongTensor(labels_val).cuda()
        x_aug = numpy2tensor(images_aug).cuda()
        y_aug = torch.LongTensor(labels_aug).cuda()

        ## val weights update
        y_pred = self.functional(wt_val, x)
        loss = self.criteria(y_pred, y)
        grad = torch.autograd.grad(loss, wt_val)
        clip_norm_(grad, self.args.gradclip)
        mean_valacc = y_pred.argmax(1).eq(y).sum().item() / len(y)
        valacc_list.append(mean_valacc)

        ## aug weights update
        y_pred_aug = self.functional(wt_aug, x_aug)
        loss_aug = self.per_criteria(y_pred_aug, y_aug)
        loss_aug = (loss_aug).mean()
        grad_aug = torch.autograd.grad(loss_aug, wt_aug)
        clip_norm_(grad_aug, self.args.gradclip)
        mean_augacc = y_pred_aug.argmax(1).eq(y_aug).sum().item() / len(y_aug)
        augacc_list.append(mean_augacc)

        g_normV = torch.sqrt(sum([torch.norm(g)**2 for g in grad]))
        g_normG = torch.sqrt(sum([torch.norm(g)**2 for g in grad_aug]))
        gradV_gradG  = sum([torch.sum(g1*g2) for g1,g2 in zip(grad, grad_aug)])

        inv_cos = -gradV_gradG/(g_normV*g_normG+1e-6)
        loss_list.append(inv_cos.item())

        loss_avg = np.mean(loss_list)
        valacc_avg = np.mean(valacc_list)
        augacc_avg = np.mean(augacc_list)

        return  loss_avg, valacc_avg, augacc_avg

    def client_test_with_calibration(self, ldr_test, wt):

        preds = []
        labels_oneh = []
        sm = nn.Softmax(dim=1)

        with torch.no_grad():
            val_loss = 0
            val_acc = 0

            ## Then we perform validation on the rest of data
            for batch_idx, (images, labels) in enumerate(ldr_test):
                x = images.cuda()
                y = labels.cuda()
                y_pred = self.functional(wt, x.cuda())
                loss = self.criteria(y_pred, y)
                val_loss += loss.mean().item()
                val_acc += y_pred.argmax(1).eq(y).sum().item() / len(y)

                ## ADDED for calibration
                # pred = y_pred.cpu().detach().numpy()
                label_oneh = torch.nn.functional.one_hot(labels, num_classes=len(ldr_test.dataset.dataset.classes))
                # label_oneh = label_oneh.cpu().detach().numpy()

                preds.extend(sm(y_pred).cpu())
                labels_oneh.extend(label_oneh)

            val_loss /= (batch_idx+1)
            val_acc /= (batch_idx+1)

            # preds = np.array(preds).flatten()
            # labels_oneh = np.array(labels_oneh).flatten()

        return  val_loss, val_acc, torch.cat(preds), torch.cat(labels_oneh)





