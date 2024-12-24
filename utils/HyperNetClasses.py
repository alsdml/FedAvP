import torch
import numpy as np
import typing
import copy
from utils.models import *

from utils.augment_utils import get_augmentation, get_lops_luniq, get_mid_magnitude
from utils.augment_utils import get_augmentation_ra, get_lops_luniq_ra 


class IdentityModel(torch.nn.Module):
    def __init__(self, base_net: torch.nn.Module, num_users, **kwargs) -> None:
        """
        Args:
            base_net: the base network
        """
        super(IdentityModel, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        #initialize parameters
        self.w_global = torch.nn.ParameterList([ torch.Tensor(p).clone().detach().requires_grad_(True) for n,p in base_net.named_parameters() if 'logit' not in n ])

    def forward(self, idx) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into functional of functorch
        """
        out = []
        for i in range(len(self.w_global)):
            out.append(self.w_global[i])
        return out 


class global_policy_meta_nomag(torch.nn.Module):
    def __init__(self, karg):
        super(global_policy_meta_nomag, self).__init__()
        len_policies = karg.l_uniq

        # init_value = torch.ones(len_policies*len_policies)*1.0
        # self.W = torch.nn.Parameter(torch.Tensor(init_value))

        H1 = karg.H1
        H2 = karg.H2
        self.dummy_embedding = torch.nn.Parameter(torch.rand(H1))
        self.L1 = torch.nn.Linear(H1,H2)
        self.L2 = torch.nn.Linear(H2,H2)
        self.L3 = torch.nn.Linear(H2,len_policies*len_policies)

        self.act = torch.nn.LeakyReLU(0.1, inplace=True)
        self.sig = torch.nn.Sigmoid()
        # self.eps = karg.eps
        # self.eps_vec = torch.Tensor(len_policies*len_policies).fill_(karg.eps)
        # self.eps_vec[0] = 0.

    def forward(self, dummy_input):
        # out = self.sig(self.W)

        out = self.act(self.L1(self.dummy_embedding))
        out = self.act(self.L2(out))
        out = self.sig(self.L3(out))
        # out = self.sig(self.L3(out))*(1.0-self.eps_vec)+self.eps_vec
        return out


class PolicyModel_meta(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super(PolicyModel_meta, self).__init__()
        karg = kwargs['args']
          
        self.policy_type = karg.policy_type
        self.server_policy = []
        # for i in range(karg.n_policies):
        modules = []
        if karg.policy_type == 0:
            modules.append(global_policy_meta_nomag(karg))
        else:
            raise Exception
        self.server_policy.append(torch.nn.Sequential(*modules))

        
        self.NP = 1#karg.n_policies
        self.act = torch.nn.Softmax()

        self.N_repeat_random=False
        self.l_mags = karg.l_mags
        self.ops_mid_magnitude = get_mid_magnitude(self.l_mags)
        self.l_ops, self.l_uniq = get_lops_luniq(karg, self.ops_mid_magnitude)
        augmentation_default, augmentation_search, augmentation_test = get_augmentation(karg)
        self.available_policies = np.arange(self.l_uniq, dtype=np.int32)[:, np.newaxis]

        self.op_names = augmentation_search.op_names
        self.unique_policy = self._get_unique_policy(self.op_names, self.l_ops, self.l_mags)
        self.eps = karg.eps 

    def forward(self) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into functional of functorch
        """
        out = []
        for i in range(self.NP):
            out.append(self.server_policy[i])

        return out

    def sample(self, images_orig, images, hist_info, sampling_weight, uniform=False):
        bs = len(images_orig)

        if uniform == True:
            samples_om = torch.randint(0,self.l_uniq*self.l_uniq,(bs,)).numpy()
        else:
            # Naive sampling 
            # samples_om = torch.multinomial(sampling_weight, num_samples=bs, replacement=True).cpu().numpy()

            # Used MetaAug paper`s sampling method
            sampling_weight_reg = (1-self.eps) * sampling_weight/sum(sampling_weight) + self.eps * 1/(self.l_uniq*self.l_uniq)
            samples_om = torch.multinomial(sampling_weight_reg, num_samples=bs, replacement=True).cpu().numpy()
            # import ipdb; ipdb.set_trace()
            

        samples_layer1 = samples_om // self.l_uniq
        samples_layer2 = samples_om % self.l_uniq

        ops_dense, mags_dense, reduce_random_mat, ops_mags_idx = self.get_dense_aug(images, repeat_random_ops=False)

        ops_1= ops_dense[samples_layer1]
        mags_1 = mags_dense[samples_layer1]
        ops_2= ops_dense[samples_layer2]
        mags_2 = mags_dense[samples_layer2]
         
        return ops_1, mags_1, ops_2, mags_2, samples_om

#
    def get_dense_aug(self, images, repeat_random_ops=False):
        ops_uniq, mags_uniq = self.unique_policy
        ops_dense = np.squeeze(ops_uniq)[self.available_policies]
        mags_dense = np.squeeze(mags_uniq)[self.available_policies]
        ops_mags_idx = self.available_policies

        nRepeat = [1] * len(self.available_policies)
        reduce_random_mat = np.eye(len(self.available_policies))

        return ops_dense, mags_dense, reduce_random_mat, ops_mags_idx

    def _get_unique_policy(self, op_names, l_ops, l_mags):
        names_modified = [op_name.split(':')[0] for op_name in op_names]
        ops_list, mags_list = [], []
        repeat_ops_idx = []
        for k_name, name in enumerate(names_modified):
            if self.ops_mid_magnitude[name] == 'random':
                repeat_ops_idx.append(k_name)
                ops_sub, mags_sub = np.array([[k_name]], dtype=np.int32), np.array([[(l_mags - 1) // 2]], dtype=np.int32)
            elif self.ops_mid_magnitude[name] is not None and self.ops_mid_magnitude[name]>=0 and self.ops_mid_magnitude[name]<=l_mags-1:
                ops_sub = k_name * np.ones([l_mags - 1, 1], dtype=np.int32)
                mags_sub = np.array([l for l in range(l_mags) if l != self.ops_mid_magnitude[name]], dtype=np.int32)[:, np.newaxis]
            elif self.ops_mid_magnitude[name] is not None and self.ops_mid_magnitude[name]<0: #or self.ops_mid_magnitude[name]>l_mags-1):
                ops_sub = k_name * np.ones([l_mags, 1], dtype=np.int32)
                mags_sub = np.arange(l_mags, dtype=np.int32)[:, np.newaxis]
            elif self.ops_mid_magnitude[name] is None:
                ops_sub, mags_sub = np.array([[k_name]], dtype=np.int32), np.array([[(l_mags - 1) // 2]], dtype=np.int32)
            else:
                raise Exception('Unrecognized middle magnitude')
            ops_list.append(ops_sub)
            mags_list.append(mags_sub)
        ops = np.concatenate(ops_list, axis=0)
        mags = np.concatenate(mags_list, axis=0)
        self.repeat_ops_idx = repeat_ops_idx
        return ops.astype(np.int32), mags.astype(np.int32)

