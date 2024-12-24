import numpy as np
import random

from utils.augment_func import AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Brightness, Sharpness, \
    Identity, Color, ShearX, ShearY, TranslateX, TranslateY, Rotate
from utils.augment_func import RandCrop, RandCutout, RandFlip

import copy
import math
from PIL import Image

CIFAR_MEANS = np.array([0.507, 0.487, 0.441], dtype=np.float32)
CIFAR_STDS = np.array([0.267, 0.256, 0.276], dtype=np.float32)

def aug_op_cifar_list():  # oeprators and their ranges
    # l = [
    #     (Identity, 0., 1.0), # 0
    #     (ShearX, -0.3, 0.3),  # 1
    #     (ShearY, -0.3, 0.3),  # 2
    #     (TranslateX, -0.45, 0.45),  # 3
    #     (TranslateY, -0.45, 0.45),  # 4
    #     (Rotate, -30., 30.),  # 5
    #     (AutoContrast, 0., 1.),  # 6
    #     (Equalize, 0., 1.),  # 7
    #     (Solarize, 0., 256.),  # 8
    #     (Posterize, 4., 8.),  # 9,
    #     (Contrast, 1.0, 1.9),  # 10
    #     (Color, 1.0, 1.9),  # 11
    #     (Brightness, 1.0, 1.9),  # 12
    #     (Sharpness, 1.0, 1.9),  # 13
    # ]
    l = [
        (Identity, 0., 1.0), # 0
        (ShearX, -0.3, 0.3),  # 1
        (ShearY, -0.3, 0.3),  # 2
        (TranslateX, -0.45, 0.45),  # 3
        (TranslateY, -0.45, 0.45),  # 4
        (Rotate, -30., 30.),  # 5
        (AutoContrast, 0., 1.),  # 6
        (Equalize, 0., 1.),  # 8
        (Solarize, 0., 256.),  # 9
        (Posterize, 4., 8.),  # 10,
        (Contrast, 0.1, 1.9),  # 11
        (Color, 0.1, 1.9),  # 12
        (Brightness, 0.1, 1.9),  # 13
        (Sharpness, 0.1, 1.9),  # 14
        (RandFlip, 0., 1.0), # 15
        (RandCutout, 0., 1.0), # 16
        (RandCrop, 0., 1.0), # 17
    ]
    names = []
    for op in l:
        info = op.__str__().split(' ')
        name = '{}:({},{}'.format(info[1], info[-2], info[-1])
        names.append(name)

    return l, names


def ra_op_cifar_list():  # oeprators and their ranges
    l = [
        (Identity, 0., 1.0), # 0
        (ShearX, -0.3, 0.3),  # 1
        (ShearY, -0.3, 0.3),  # 2
        (TranslateX, -0.45, 0.45),  # 3
        (TranslateY, -0.45, 0.45),  # 4
        (Rotate, -30., 30.),  # 5
        (AutoContrast, 0., 1.),  # 6
        (Equalize, 0., 1.),  # 7
        (Solarize, 0., 256.),  # 8
        (Posterize, 4., 8.),  # 9,
        (Contrast, 1.0, 1.9),  # 10
        (Color, 1.0, 1.9),  # 11
        (Brightness, 1.0, 1.9),  # 12
        (Sharpness, 1.0, 1.9),  # 13
    ]
    names = []
    for op in l:
        info = op.__str__().split(' ')
        name = '{}:({},{}'.format(info[1], info[-2], info[-1])
        names.append(name)

    return l, names

def get_augmentation(args):
    if 'cifar' in args.dataset:
        augmentation_default = DataAugmentation(dataset=args.dataset, args=args,
                                        ops_list=(None, None),
                                        default_pre_aug=None,
                                        default_post_aug=[RandCrop,
                                                          RandFlip,
                                                          RandCutout])

        augmentation_search = DataAugmentation(dataset=args.dataset, args=args,
                                                ops_list=aug_op_cifar_list(),
                                                default_pre_aug=None,
                                                default_post_aug=None)

        augmentation_test = DataAugmentation(dataset=args.dataset, args=args,
                                             ops_list=(None, None),
                                             default_pre_aug=None,
                                             default_post_aug=None)
    
    return augmentation_default, augmentation_search, augmentation_test


def get_augmentation_independent_mag(args):
    if 'cifar' in args.dataset:
        augmentation_default = DataAugmentation_independent_mag(dataset=args.dataset, args=args,
                                        ops_list=(None, None),
                                        default_pre_aug=None,
                                        default_post_aug=[RandCrop,
                                                          RandFlip,
                                                          RandCutout])

        augmentation_search = DataAugmentation_independent_mag(dataset=args.dataset, args=args,
                                                ops_list=aug_op_cifar_list(),
                                                default_pre_aug=None,
                                                default_post_aug=None)

        augmentation_test = DataAugmentation_independent_mag(dataset=args.dataset, args=args,
                                             ops_list=(None, None),
                                             default_pre_aug=None,
                                             default_post_aug=None)
    
    return augmentation_default, augmentation_search, augmentation_test

def get_augmentation_fix_mag(args):
    if 'cifar' in args.dataset:
        augmentation_default = DataAugmentation_fix_mag(dataset=args.dataset, args=args,
                                        ops_list=(None, None),
                                        default_pre_aug=None,
                                        default_post_aug=[RandCrop,
                                                          RandFlip,
                                                          RandCutout])

        augmentation_search = DataAugmentation_fix_mag(dataset=args.dataset, args=args,
                                                ops_list=aug_op_cifar_list(),
                                                default_pre_aug=None,
                                                default_post_aug=None)

        augmentation_test = DataAugmentation_fix_mag(dataset=args.dataset, args=args,
                                             ops_list=(None, None),
                                             default_pre_aug=None,
                                             default_post_aug=None)
    
    return augmentation_default, augmentation_search, augmentation_test

def get_augmentation_ra(args):
    if 'cifar' in args.dataset:
        augmentation_default = DataAugmentation_ra_mag(dataset=args.dataset, args=args,
                                        ops_list=(None, None),
                                        default_pre_aug=None,
                                        default_post_aug=[RandCrop,
                                                          RandFlip,
                                                          RandCutout])
        if args.post_aug == 0:
            augmentation_search = DataAugmentation_ra_mag(dataset=args.dataset, args=args,
                                                    ops_list=ra_op_cifar_list(),
                                                    default_pre_aug=None,
                                                    default_post_aug=None)
        else:
            augmentation_search = DataAugmentation_ra_mag(dataset=args.dataset, args=args,
                                                    ops_list=ra_op_cifar_list(),
                                                    default_pre_aug=None,
                                                    default_post_aug=[RandCrop,
                                                                      RandFlip,
                                                                      ])

        augmentation_test = DataAugmentation_ra_mag(dataset=args.dataset, args=args,
                                             ops_list=(None, None),
                                             default_pre_aug=None,
                                             default_post_aug=None)
    
    return augmentation_default, augmentation_search, augmentation_test

def get_lops_luniq(args, ops_mid_magnitude):
    if 'cifar' in args.dataset:
        _, op_names = aug_op_cifar_list()
    else:
        raise Exception('Unknown dataset ={}'.format(args.dataset))

    names_modified = [op_name.split(':')[0] for op_name in op_names]
    l_ops = len(op_names)
    l_uniq = 0
    for k_name, name in enumerate(names_modified):
        mid_mag = ops_mid_magnitude[name]

        if mid_mag == 'random':
           l_uniq += 1 # The op is a random op, however we only sample one op
        elif mid_mag is not None and mid_mag >=0 and mid_mag <= args.l_mags-1:
            l_uniq += args.l_mags-1
        elif mid_mag is not None and mid_mag == -1: # magnitude==-1 means all l_mags are independnt policies; or mid_mag > args.l_mags-1)
            l_uniq += args.l_mags
        elif mid_mag is None:
            l_uniq += 1
        else:
            raise Exception('mid_mag = {} is invalid'.format(mid_mag))
    return l_ops, l_uniq

def get_lops_luniq_ra(args, ops_mid_magnitude):
    if 'cifar' in args.dataset:
        _, op_names = ra_op_cifar_list()
    else:
        raise Exception('Unknown dataset ={}'.format(args.dataset))

    names_modified = [op_name.split(':')[0] for op_name in op_names]
    l_ops = len(op_names)
    l_uniq = 0
    for k_name, name in enumerate(names_modified):
        mid_mag = ops_mid_magnitude[name]

        if mid_mag == 'random':
           l_uniq += 1 # The op is a random op, however we only sample one op
        elif mid_mag is not None and mid_mag >=0 and mid_mag <= args.l_mags-1:
            l_uniq += args.l_mags-1
        elif mid_mag is not None and mid_mag == -1: # magnitude==-1 means all l_mags are independnt policies; or mid_mag > args.l_mags-1)
            l_uniq += args.l_mags
        elif mid_mag is None:
            l_uniq += 1
        else:
            raise Exception('mid_mag = {} is invalid'.format(mid_mag))
    return l_ops, l_uniq

def get_mid_magnitude(l_mags):

    if l_mags >= 1:
        ops_mid_magnitude = {'Identity': None,
                            'ShearX': (l_mags - 1) // 2,
                            'ShearY': (l_mags - 1) // 2,
                            'TranslateX': (l_mags - 1) // 2,
                            'TranslateY': (l_mags - 1) // 2,
                            'Rotate': (l_mags - 1) // 2,
                            'AutoContrast': None,
                            'Invert': None,
                            'Equalize': None,
                            'Solarize': l_mags - 1,
                            'Posterize': l_mags - 1,
                            'Contrast': (l_mags - 1) // 2,
                            'Color': (l_mags - 1) // 2,
                            'Brightness': (l_mags - 1) // 2,
                            'Sharpness': (l_mags - 1) // 2,
                            # 'Contrast': 0, # upper mag only
                            # 'Color': 0,
                            # 'Brightness': 0,
                            # 'Sharpness': 0,
                            'RandFlip': 'random',
                            'RandCutout': 'random',
                            'RandCutout60': 'random',
                            'RandCrop': 'random',
                            'RandResizeCrop_imagenet': 'random',
                            }
    else:
        # ops_mid_magnitude = {'Identity': None,
        #                     'ShearX': None,
        #                     'ShearY': None,
        #                     'TranslateX': None,
        #                     'TranslateY': None,
        #                     'Rotate': None,
        #                     'AutoContrast': None,
        #                     'Invert': None,
        #                     'Equalize': None,
        #                     'Solarize': None,
        #                     'Posterize': None,
        #                     'Contrast': None,
        #                     'Color': None,
        #                     'Brightness': None,
        #                     'Sharpness': None,
        #                     'RandFlip': None,
        #                     'RandCutout': None,
        #                     'RandCutout60': None,
        #                     'RandCrop': None,
        #                     'RandResizeCrop_imagenet': None,
        #                     }
        ops_mid_magnitude = {'Identity': None,
                            'ShearX': 'random',
                            'ShearY': 'random',
                            'TranslateX': 'random',
                            'TranslateY': 'random',
                            'Rotate': 'random',
                            'AutoContrast': None,
                            'Invert': None,
                            'Equalize': None,
                            'Solarize': 'random',
                            'Posterize': 'random',
                            'Contrast': 'random',
                            'Color': 'random',
                            'Brightness': 'random',
                            'Sharpness': 'random',
                            'RandFlip': 'random',
                            'RandCutout': 'random',
                            'RandCutout60': 'random',
                            'RandCrop': 'random',
                            'RandResizeCrop_imagenet': 'random',
                            }
    return ops_mid_magnitude



class Mod_PrefetchGenerator():
    def __init__(self, search_ds, val_ds, inter_classes, search_bs=8, val_bs=64):
        self.search_ds = search_ds
        self.val_ds = val_ds
        self.inter_classes = inter_classes
        self.search_bs = search_bs
        self.val_bs = val_bs

    @staticmethod
    def sample_label_and_batch(dataset, bs, inter_classes, MAX_iterations=100):
        for k in range(MAX_iterations):
            try:
                lab = random.sample(inter_classes,1)[0]
                imgs, labs = dataset.sample_labeled_data_batch(lab, bs)
            except:
                print('Insufficient data in a single class, try {}/{}'.format(k, MAX_iterations))
                continue
            return lab, imgs, labs
        raise Exception('Maximum number of iteration {} reached'.format(MAX_iterations))

    def next(self):
        images_val, labels_val, images_train, labels_train = [], [], [], []
        for _ in range(self.search_bs):
            lab, imgs_val, labs_val = Mod_PrefetchGenerator.sample_label_and_batch(self.val_ds, self.val_bs, self.inter_classes)
            imgs_train, labs_train = self.search_ds.sample_labeled_data_batch(lab, 1)
            images_val.append(imgs_val)
            labels_val.append(labs_val)
            images_train.append(imgs_train)
            labels_train.append(labs_train)
        return (images_val, labels_val, images_train, labels_train)

class Mod_DataGenerator():
    def __init__(self,
                 data,
                 labels,
                ):

        self._data = data
        self.data = self._data # initially without calling augment, the output data is not augmented
        self.labels = labels

    def reset_augment(self):
        self.data = self._data

    def sample_labeled_data_batch(self, label, bs):
        # suffle indices every time
        indices = np.arange(len(self._data))
        np.random.shuffle(indices)
        if isinstance(self.labels, list):
            labels = [self.labels[k] for k in indices]
        else:
            labels = self.labels[indices]
        matched_labels = np.array(labels) == int(label)
        matched_indices = [id for id, isMatched in enumerate(matched_labels) if isMatched]

        if len(matched_indices) - bs >=0:
            if len(matched_indices) - bs == 0:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, len(matched_indices)-bs)
            batch_indices = matched_indices[start_idx:start_idx + bs]
        else:
            #print('Not enough matched data, required {}, but got {} instead'.format(bs, len(matched_indices)))
            batch_indices = matched_indices

        data_indices = indices[batch_indices]
        return [self.data[k] for k in data_indices], np.array([self.labels[k] for k in data_indices], dtype=self.labels[0].dtype)

class DataAugmentation(object):
    def __init__(self, dataset, args, ops_list=None, default_pre_aug=None, default_post_aug=None):
        self.ops, self.op_names = ops_list
        self.default_pre_aug = default_pre_aug
        self.default_post_aug = default_post_aug
        self.dataset = dataset
        # self.fixed_mag_config = args.fixed_mag_config
        # self.fixed_mag = args.fixed_mag
        self.l_mags = args.l_mags 
        if 'cifar' in self.dataset:
            self.image_shape = (32, 32, 3)
        else:
            raise Exception('Unrecognized dataset')

        if 'cifar100' == self.dataset:
            self.CIFAR_MEANS = np.array([0.507, 0.487, 0.441], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.267, 0.256, 0.276], dtype=np.float32)
        elif 'cifar10' == self.dataset:
            self.CIFAR_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # def sequantially_augment(self, idx, img_, op_idxs, mags): # for debug 
    def sequantially_augment(self, args):
        idx, img_, op_idxs, mags, aug_finish = args
        assert img_.dtype == np.uint8, 'Input images should be unporocessed, should stay in np.uint8'
        img = copy.deepcopy(img_)
        pil_img = Image.fromarray(img)  # Convert to PIL.Image
        if self.default_pre_aug is not None:
            for op in self.default_pre_aug:
                pil_img = op(pil_img)
        if self.ops is not None:
            for op_idx, mag in zip(op_idxs, mags):
                op, minval, maxval = self.ops[op_idx]
                # if self.fixed_mag_config:
                #     op_name = self.op_names[op_idx].split(':')[0]
                #     if op_name in ['Posterize', 'Solarize']:
                #         mag = (1.0-self.fixed_mag)
                #     elif op_name in['Contrast','Color','Brightness','Sharpness']:
                #         mag = (self.fixed_mag)
                #     else:
                #         mag = (self.fixed_mag)*0.5 + 0.5 # Assume config.random_mirror is True
                #     # elif op_name in ['AutoContrast', 'Invert','Equalize','RandFlip','RandCutout','RandCrop']:
                #     mag = mag * (maxval - minval) + minval
                # else: 
                if self.l_mags >= 1:
                    assert mag > -1e-5 and mag < 1. + 1e-5, 'magnitudes should be in the range of (0., 1.)'
                    mag = mag * (maxval - minval) + minval
                else:
                    mag = random.random() * (maxval - minval) + minval

                pil_img = op(pil_img, mag)
        if self.default_post_aug is not None and self.use_post_aug:
            for op in self.default_post_aug:
                pil_img = op(pil_img, None)
        if 'cifar' in self.dataset:
            img = np.asarray(pil_img, dtype=np.uint8)
            return idx, img
        else:
            raise Exception

    def postprocessing_standardization(self, pil_img):
        x = np.asarray(pil_img, dtype=np.float32) / 255.
        if 'cifar' in self.dataset:
            x = (x - self.CIFAR_MEANS) / self.CIFAR_STDS
        else:
            raise Exception('Unrecoginized dataset')
        return x

    def __call__(self, images, labels, samples_op, samples_mag, use_post_aug, pool=None, chunksize=None, aug_finish=True):

        self.use_post_aug = use_post_aug
        self.batch_len = len(labels)
        if aug_finish:
            aug_imgs = np.empty([self.batch_len, *self.image_shape], dtype=np.float32)
        else:
            aug_imgs = [None]*self.batch_len

        # aug_results = self.sequantially_augment(range(1), images[3], samples_op[3], samples_mag[3]) # for debug 
        
        aug_results = pool.imap_unordered(self.sequantially_augment,
                                          zip(range(self.batch_len), images, samples_op, samples_mag, [aug_finish]*self.batch_len),
                                          chunksize=math.ceil(float(self.batch_len) / float(pool._processes)) if chunksize is None else chunksize)

        for idx, img in aug_results:
            aug_imgs[idx] = img

        if aug_finish:
            aug_imgs = self.postprocessing_standardization(aug_imgs)

        return aug_imgs, labels



class DataAugmentation_independent_mag(object):
    def __init__(self, dataset, args, ops_list=None, default_pre_aug=None, default_post_aug=None):
        self.ops, self.op_names = ops_list
        self.default_pre_aug = default_pre_aug
        self.default_post_aug = default_post_aug
        self.dataset = dataset
        # self.fixed_mag_config = args.fixed_mag_config
        # self.fixed_mag = args.fixed_mag
        self.l_mags = args.l_mags 
        if 'cifar' in self.dataset:
            self.image_shape = (32, 32, 3)
        else:
            raise Exception('Unrecognized dataset')

        if 'cifar100' == self.dataset:
            self.CIFAR_MEANS = np.array([0.507, 0.487, 0.441], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.267, 0.256, 0.276], dtype=np.float32)
        elif 'cifar10' == self.dataset:
            self.CIFAR_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # def sequantially_augment(self, idx, img_, op_idxs, mags): # for debug 
    def sequantially_augment(self, args):
        idx, img_, op_idxs, mags, aug_finish = args
        assert img_.dtype == np.uint8, 'Input images should be unporocessed, should stay in np.uint8'
        img = copy.deepcopy(img_)
        pil_img = Image.fromarray(img)  # Convert to PIL.Image
        if self.default_pre_aug is not None:
            for op in self.default_pre_aug:
                pil_img = op(pil_img)
        if self.ops is not None:
            for op_idx, mag_ in zip(op_idxs, mags):
                op, minval, maxval = self.ops[op_idx]
                op_name = self.op_names[op_idx].split(':')[0]
                if op_name in ['Posterize', 'Solarize']:
                    mag = (1.0-mag_)
                else:
                    mag = (mag_)*0.5 + 0.5 # Assume config.random_mirror is True
                # elif op_name in ['AutoContrast', 'Invert','Equalize','RandFlip','RandCutout','RandCrop']: # These operations dont care magnitude
                mag = mag * (maxval - minval) + minval

                pil_img = op(pil_img, mag)
        if self.default_post_aug is not None and self.use_post_aug:
            for op in self.default_post_aug:
                pil_img = op(pil_img, None)
        if 'cifar' in self.dataset:
            img = np.asarray(pil_img, dtype=np.uint8)
            return idx, img
        else:
            raise Exception

    def postprocessing_standardization(self, pil_img):
        x = np.asarray(pil_img, dtype=np.float32) / 255.
        if 'cifar' in self.dataset:
            x = (x - self.CIFAR_MEANS) / self.CIFAR_STDS
        else:
            raise Exception('Unrecoginized dataset')
        return x

    def __call__(self, images, labels, samples_op, samples_mag, use_post_aug, pool=None, chunksize=None, aug_finish=True):

        self.use_post_aug = use_post_aug
        self.batch_len = len(labels)
        if aug_finish:
            aug_imgs = np.empty([self.batch_len, *self.image_shape], dtype=np.float32)
        else:
            aug_imgs = [None]*self.batch_len

        aug_results = pool.imap_unordered(self.sequantially_augment,
                                          zip(range(self.batch_len), images, samples_op, samples_mag, [aug_finish]*self.batch_len),
                                          chunksize=math.ceil(float(self.batch_len) / float(pool._processes)) if chunksize is None else chunksize)

        for idx, img in aug_results:
            aug_imgs[idx] = img

        if aug_finish:
            aug_imgs = self.postprocessing_standardization(aug_imgs)

        return aug_imgs, labels


class DataAugmentation_fix_mag(object):
    def __init__(self, dataset, args, ops_list=None, default_pre_aug=None, default_post_aug=None):
        self.ops, self.op_names = ops_list
        self.default_pre_aug = default_pre_aug
        self.default_post_aug = default_post_aug
        self.dataset = dataset
        # self.fixed_mag_config = args.fixed_mag_config
        self.fixed_mag = args.fixed_mag
        self.l_mags = args.l_mags 
        if 'cifar' in self.dataset:
            self.image_shape = (32, 32, 3)
        else:
            raise Exception('Unrecognized dataset')

        if 'cifar100' == self.dataset:
            self.CIFAR_MEANS = np.array([0.507, 0.487, 0.441], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.267, 0.256, 0.276], dtype=np.float32)
        elif 'cifar10' == self.dataset:
            self.CIFAR_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def sequantially_augment(self, args):
        idx, img_, op_idxs, mags, aug_finish = args
        assert img_.dtype == np.uint8, 'Input images should be unporocessed, should stay in np.uint8'
        img = copy.deepcopy(img_)
        pil_img = Image.fromarray(img)  # Convert to PIL.Image
        if self.default_pre_aug is not None:
            for op in self.default_pre_aug:
                pil_img = op(pil_img)
        if self.ops is not None:
            for op_idx, mag_ in zip(op_idxs, mags):
                op, minval, maxval = self.ops[op_idx]
                op_name = self.op_names[op_idx].split(':')[0]
                if op_name in ['Posterize', 'Solarize']:
                    mag = (1.0-self.fixed_mag)
                else:
                    mag = (self.fixed_mag)*0.5 + 0.5 # Assume config.random_mirror is True
                # elif op_name in ['AutoContrast', 'Invert','Equalize','RandFlip','RandCutout','RandCrop']: # These operations dont care magnitude
                mag = mag * (maxval - minval) + minval

                pil_img = op(pil_img, mag)
        if self.default_post_aug is not None and self.use_post_aug:
            for op in self.default_post_aug:
                pil_img = op(pil_img, None)
        if 'cifar' in self.dataset:
            img = np.asarray(pil_img, dtype=np.uint8)
            return idx, img
        else:
            raise Exception

    def postprocessing_standardization(self, pil_img):
        x = np.asarray(pil_img, dtype=np.float32) / 255.
        if 'cifar' in self.dataset:
            x = (x - self.CIFAR_MEANS) / self.CIFAR_STDS
        else:
            raise Exception('Unrecoginized dataset')
        return x

    def __call__(self, images, labels, samples_op, samples_mag, use_post_aug, pool=None, chunksize=None, aug_finish=True):

        self.use_post_aug = use_post_aug
        self.batch_len = len(labels)
        if aug_finish:
            aug_imgs = np.empty([self.batch_len, *self.image_shape], dtype=np.float32)
        else:
            aug_imgs = [None]*self.batch_len
        
        aug_results = pool.imap_unordered(self.sequantially_augment,
                                          zip(range(self.batch_len), images, samples_op, samples_mag, [aug_finish]*self.batch_len),
                                          chunksize=math.ceil(float(self.batch_len) / float(pool._processes)) if chunksize is None else chunksize)

        for idx, img in aug_results:
            aug_imgs[idx] = img

        if aug_finish:
            aug_imgs = self.postprocessing_standardization(aug_imgs)

        return aug_imgs, labels
    


class DataAugmentation_ra_mag(object):
    def __init__(self, dataset, args, ops_list=None, default_pre_aug=None, default_post_aug=None):
        self.ops, self.op_names = ops_list
        self.default_pre_aug = default_pre_aug
        self.default_post_aug = default_post_aug
        self.dataset = dataset
        # self.fixed_mag_config = args.fixed_mag_config
        self.fixed_mag = args.fixed_mag
        self.l_mags = args.l_mags 
        if 'cifar' in self.dataset:
            self.image_shape = (32, 32, 3)
        else:
            raise Exception('Unrecognized dataset')

        if 'cifar100' == self.dataset:
            self.CIFAR_MEANS = np.array([0.507, 0.487, 0.441], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.267, 0.256, 0.276], dtype=np.float32)
        elif 'cifar10' == self.dataset:
            self.CIFAR_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.CIFAR_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def sequantially_augment(self, args):
        idx, img_, op_idxs, mags, aug_finish = args
        assert img_.dtype == np.uint8, 'Input images should be unporocessed, should stay in np.uint8'
        img = copy.deepcopy(img_)
        pil_img = Image.fromarray(img)  # Convert to PIL.Image
        if self.default_pre_aug is not None:
            for op in self.default_pre_aug:
                pil_img = op(pil_img)
        if self.ops is not None:
            for op_idx, mag_ in zip(op_idxs, mags):
                op, minval, maxval = self.ops[op_idx]
                op_name = self.op_names[op_idx].split(':')[0]

                if op_name in ['Posterize', 'Solarize']:
                    mag = (1.0-self.fixed_mag)
                elif op_name in['Contrast','Color','Brightness','Sharpness']:
                    mag = (self.fixed_mag)
                else:
                    mag = (self.fixed_mag)*0.5 + 0.5 # Assume config.random_mirror is True
                # elif op_name in ['AutoContrast', 'Invert','Equalize','RandFlip','RandCutout','RandCrop']: # These operations dont care magnitude
                mag = mag * (maxval - minval) + minval

                pil_img = op(pil_img, mag)
        if self.default_post_aug is not None and self.use_post_aug:
            for op in self.default_post_aug:
                pil_img = op(pil_img, None)
        if 'cifar' in self.dataset:
            img = np.asarray(pil_img, dtype=np.uint8)
            return idx, img
        else:
            raise Exception

    def postprocessing_standardization(self, pil_img):
        x = np.asarray(pil_img, dtype=np.float32) / 255.
        if 'cifar' in self.dataset:
            x = (x - self.CIFAR_MEANS) / self.CIFAR_STDS
        else:
            raise Exception('Unrecoginized dataset')
        return x

    def __call__(self, images, labels, samples_op, samples_mag, use_post_aug, pool=None, chunksize=None, aug_finish=True):

        self.use_post_aug = use_post_aug
        self.batch_len = len(labels)
        if aug_finish:
            aug_imgs = np.empty([self.batch_len, *self.image_shape], dtype=np.float32)
        else:
            aug_imgs = [None]*self.batch_len
        
        aug_results = pool.imap_unordered(self.sequantially_augment,
                                          zip(range(self.batch_len), images, samples_op, samples_mag, [aug_finish]*self.batch_len),
                                          chunksize=math.ceil(float(self.batch_len) / float(pool._processes)) if chunksize is None else chunksize)

        for idx, img in aug_results:
            aug_imgs[idx] = img

        if aug_finish:
            aug_imgs = self.postprocessing_standardization(aug_imgs)

        return aug_imgs, labels
