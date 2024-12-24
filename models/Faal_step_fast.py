import torch
from models.BaseModel import BaseModel
from utils.models import CNNCifar
from utils.HyperNetClasses import IdentityModel, PolicyModel_meta
from functorch import make_functional
import numpy as np
from utils.train_utils import clip_norm_, repeat, clip_norm_coef
import uuid
import torchvision


def numpy2tensor(batch_input):
    L = batch_input.shape[0]
    output = torch.Tensor(L, 3, 32, 32)
    TT = torchvision.transforms.ToTensor()
    for i in range(L):
        output[i] = TT(batch_input[i])
    return output

def tensor2numpy(images, labels):
    images = [[e.squeeze(0).numpy() for e in images]]
    labels = [labels.numpy().flatten()]
    return images, labels 


class Faal_step_fast(BaseModel):
    def __init__(self, args, num_classes):
        args.beta = 0
        args.aggw = 0
        args.client_lr = 0
        args.gradclip = 10
        args.use_cyclical_beta = False

        args.uuid = str(uuid.uuid1())[:8]
        args.exp = f'{getattr(args, "pre", "")}_{args.algorithm}_{args.dataset}_{args.alpha}_mfrac:{args.frac_m}('
        args.exp += f'replr:{args.policy_lr}_'
        args.exp += f'w_ilr:{args.w_inner_lr}_'
        args.exp += f'p_ilr:{args.p_inner_lr}_'
        args.exp += f'pwd:{args.pwd}_'
        args.exp += f'ptype:{args.policy_type}_'
        args.exp += f'expns:{args.expansion}_'
        args.exp += f'pclip:{args.gradclip_policy}_'
        args.exp += f'fo:{args.first_order}_' 
        args.exp += f'sn:{args.score_norm}_' 
        args.exp += f'vbs:{args.val_bs}_'
        args.exp += f'eps:{args.eps}_'
        args.exp += f'H1:{args.H1}_'
        args.exp += f'H2:{args.H2}_'
        args.exp += f"u:{args.num_users}_ou:{args.ood_users}_"
        args.exp += f')'+f'_{args.uuid}'
        print(args.exp)

        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifar(self.args.num_classes)

        self.model = self.model.cuda()
        self.hpnet = IdentityModel(self.model, self.args.num_users)
        self.policynet = PolicyModel_meta(args=args)

        self.hpnet_policynet = []
        self.hpnet_policynet.append(IdentityModel(self.policynet.server_policy[0], self.args.num_users))
        
        self.pserver_optimizer = []
        stage_optimizer = torch.optim.SGD(self.hpnet_policynet[0].parameters(), lr=args.policy_lr, weight_decay=args.pwd)
        self.pserver_optimizer.append(stage_optimizer)
        
        ops_dense, mags_dense, reduce_random_mat, ops_mags_idx = self.policynet.get_dense_aug(None,False)
        self.len_op = len(ops_dense)

        self.functional, self.gparam = make_functional(self.model)
        self.pfunctional, self.pparam = make_functional(self.policynet.server_policy[0])
        # self.adapted_policy = torch.zeros(args.num_users+args.ood_users, args.n_policies, len(ops_dense)*len(ops_dense))
        self.adapted_policy = torch.zeros(args.num_users+args.ood_users, 1, len(ops_dense)*len(ops_dense))

        self.server_optimizer = torch.optim.SGD(self.hpnet.w_global.parameters(), lr=1.0, momentum=.0, weight_decay=0) 


    def client_update(self, ldr_train, wt, p_local, aug_search, aug_test, pool, args, stage, hist_info, idx, w_local=None, round=None, local_epoch=None):
        loss_list = []
        ploss_list = []
        valacc_list = []
        augacc_list = []
        lgrad_list = []
        weights_2_dict = {}

        ops_dense, mags_dense, reduce_random_mat, ops_mags_idx = self.policynet.get_dense_aug(None,False)

        wt_model = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in wt]
        wt_policy = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in p_local]

        eval = ldr_train.dataset.eval

        inner_steps = args.inner_steps
        inner_steps = len(ldr_train)
        for i in range(inner_steps):
            wt_policy = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in wt_policy]
            weights_2 = self.pfunctional(wt_policy, hist_info).cuda()
            if i in [0, inner_steps-1]:
                weights_2_dict[f'{idx}_{round}_{local_epoch}_{i}'] = weights_2.detach().reshape(self.len_op,self.len_op).clone().cpu().numpy()
           
            ## aug grad
            images_aug, labels_aug, _ = next(ldr_train.__iter__())
            images_aug = [images_aug[i].unsqueeze(0) for i in range(images_aug.size(0))]
            images_aug, labels_aug = tensor2numpy(images_aug, labels_aug)

            if not eval and (i%5 == 0):
                expansion = args.expansion
            else:
                expansion = 1

            sample_size = len(labels_aug[0]) * args.expansion 
            dummy_images = [None] * sample_size
            ops_layer1, mags_layer1, ops_layer2, mags_layer2, sample_om = self.policynet.sample(dummy_images, dummy_images, hist_info, weights_2)
            ops_k, mags_k = [], []
            ops_k.append(ops_layer1)
            mags_k.append(mags_layer1)
            ops_k.append(ops_layer2)
            mags_k.append(mags_layer2)
            ops_k = np.concatenate(ops_k, axis=1)  
            mags_k = np.concatenate(mags_k, axis=1) 

            images_aug, labels_aug = aug_search(repeat(sum(images_aug,[]), args.expansion, axis=0), repeat(np.concatenate(labels_aug, axis=0), args.expansion, axis=0),
                                                    ops_k, mags_k.astype(np.float32)/float(args.l_mags-1),
                                                    use_post_aug=False, pool=pool, chunksize=None)
            images_aug = np.reshape(images_aug, [sample_size, 32, 32, 3])
            labels_aug = np.reshape(labels_aug, [sample_size])
            weights_aug = weights_2[sample_om]

            x_aug = numpy2tensor(images_aug).cuda()
            y_aug = torch.LongTensor(labels_aug).cuda()

            y_pred = self.functional(wt_model, x_aug)
            loss_aug = self.per_criteria(y_pred, y_aug)
            if self.args.score_norm == 1:
                normalized_weights = weights_aug/sum(weights_aug)
                loss_aug = (loss_aug*normalized_weights).sum()
            else: 
                loss_aug = (loss_aug*weights_aug).mean()
            mean_augacc = y_pred.argmax(1).eq(y_aug).sum().item() / len(y_aug)
            augacc_list.append(mean_augacc)

            make_graph = not eval and (i%5 == 0)
            grad_aug = torch.autograd.grad(loss_aug, wt_model, create_graph=make_graph)
            coef = clip_norm_coef(grad_aug, self.args.gradclip)
            wt_model = [w - coef * self.args.w_inner_lr * g for w, g in zip(wt_model, grad_aug)]
            loss_list.append(loss_aug.item())

            if not eval and (i%5 == 0):
                ## val grad
                images_val, labels_val, _ = next(ldr_train.__iter__())
                images_val = [images_val[i].unsqueeze(0) for i in range(images_val.size(0))]
                images_val, labels_val = tensor2numpy(images_val, labels_val)
                val_bs = len(labels_val[0])
                images_val, labels_val = aug_test(sum(images_val,[]), np.concatenate(labels_val),
                                                    np.array([[0]]*val_bs, dtype=np.int32),
                                                    np.array([[0]]*val_bs, dtype=np.float32)/float(args.l_mags-1),
                                                    use_post_aug=True, pool=pool, chunksize=None)
                images_val = np.reshape(images_val, [val_bs, 32, 32, 3])
                labels_val = np.reshape(labels_val, [val_bs])
                x = numpy2tensor(images_val).cuda()
                y = torch.LongTensor(labels_val).cuda()
                y_pred = self.functional(wt_model, x)
                loss = self.criteria(y_pred, y)
                grad_val = torch.autograd.grad(loss, wt_model)
                mean_valacc = y_pred.argmax(1).eq(y).sum().item() / len(y)
                valacc_list.append(mean_valacc)   


            if not eval and (i%5 == 0):
                # policy grad
                if self.args.gradclip_policy <= 3.0:
                    coef_aug = clip_norm_coef(grad_aug, self.args.gradclip_policy)
                    coef_val = clip_norm_coef(grad_val, self.args.gradclip_policy)
                    inner_product  = sum([torch.sum(coef_aug*coef_val*g1*g2) for g1,g2 in zip(grad_aug, grad_val)])
                else: # ablation
                    inner_product  = sum([torch.sum(g1*g2) for g1,g2 in zip(grad_aug, grad_val)])

                ploss_list.append(inner_product.item())

                grad_policy = torch.autograd.grad(-inner_product, wt_policy)
                if self.args.first_order == 1:
                    wt_model = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in wt_model] # free graph
                clip_norm_(grad_policy, self.args.gradclip)
                 
                wt_policy = [w - self.args.p_inner_lr * g for w, g in zip(wt_policy, grad_policy)]

        loss_avg = np.mean(loss_list) 
        ploss_avg = np.mean(ploss_list) if len(ploss_list)>0 else torch.Tensor([0]) 
        valacc_avg = np.mean(valacc_list) if len(valacc_list)>0 else 0 
        augacc_avg = np.mean(augacc_list)
        return wt_model, wt_policy, loss_avg, ploss_avg, valacc_avg, augacc_avg, weights_2_dict
    

    def server_policy_aggregation(self, user_idxs, users_datasize, collection):
        stage = 0
        collected_weights = collection[0]

        weights_size = []
        for idx in user_idxs:
            weights_size.append(users_datasize[idx])

        if sum(weights_size) != 0:
            weights = torch.Tensor(weights_size) / sum(weights_size)

            for i, idx in enumerate(user_idxs):
                w_local = self.hpnet_policynet[stage](idx)
                delta_theta = [ torch.Tensor((wg - wl).data).detach().clone() for wg , wl in zip(w_local, collected_weights[i])]
                hnet_grads = torch.autograd.grad(w_local, self.hpnet_policynet[stage].parameters(), delta_theta, allow_unused=True)

                for (name, p), g in zip(self.hpnet_policynet[stage].named_parameters(), hnet_grads):
                    if p.grad == None:
                        p.grad = torch.zeros_like(p)
                    if g == None:
                        g = torch.zeros_like(p)
                    p.grad = p.grad + g * weights[i]

            torch.nn.utils.clip_grad_norm_(self.hpnet_policynet[stage].parameters(), 10)

            self.pserver_optimizer[stage].step()
            self.pserver_optimizer[stage].zero_grad()


    def server_aggregation(self, user_idxs, users_datasize, collection):

        collected_weights = collection[0]

        weights_size = []
        for idx in user_idxs:
            weights_size.append(users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        for i, idx in enumerate(user_idxs):
            w_local = self.hpnet(idx)
            delta_theta = [ torch.Tensor((wg - wl).data).detach().clone() for wg , wl in zip(w_local, collected_weights[i])]
            hnet_grads = torch.autograd.grad(w_local, self.hpnet.parameters(), delta_theta, allow_unused=True)

            for (name, p), g in zip(self.hpnet.named_parameters(), hnet_grads):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                if g == None:
                    g = torch.zeros_like(p)
                p.grad = p.grad + g * weights[i]

        torch.nn.utils.clip_grad_norm_(self.hpnet.parameters(), 10)

        self.server_optimizer.step()
        self.server_optimizer.zero_grad()


    def client_adapt(self, ldr_train, wt, p_local, aug_search, aug_test, pool, args, stage, hist_info, idx, w_local=None):

        ops_dense, mags_dense, reduce_random_mat, ops_mags_idx = self.policynet.get_dense_aug(None,False)

        wt_model = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in wt]
        wt_policy = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in p_local]
 
        if args.policy_type in [0, 1, -1]:
            weights_2 = self.pfunctional(wt_policy, hist_info).cuda()
        self.adapted_policy[idx][stage] = weights_2.clone().detach()

        return wt

