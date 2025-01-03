import random
import math

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import os

def find_folders(root_dir, lastname):
    matching_folders = []
    for root, dirs, files in os.walk(root_dir):
        for folder in dirs:
            if folder.endswith(lastname):
                matching_folders.append(os.path.join(root, folder))
    assert len(matching_folders) == 1
    return matching_folders[0]

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    # if config.random_mirror and random.random() > 0.5:
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    # if config.random_mirror and random.random() > 0.5:
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    # if config.random_mirror and random.random() > 0.5:
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    # if config.random_mirror and random.random() > 0.5:
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    # if config.random_mirror and random.random() > 0.5:
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    # if config.random_mirror and random.random() > 0.5:
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    # if config.random_mirror and random.random() > 0.5:
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return Cutout_default(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    # x0 = np.random.uniform(w)
    # y0 = np.random.uniform(h)
    x0 = random.uniform(0, w)
    y0 = random.uniform(0, h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    # color = (125, 123, 114)
    color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f

def mean_pad_randcrop(img, v):
    # v: Pad with mean value=[125, 123, 114] by v pixels on each side and then take random crop
    assert v <= 10, 'The maximum shift should be less then 10'
    padded_size = (img.size[0] + 2*v, img.size[1] + 2*v)
    new_img = PIL.Image.new('RGB', padded_size, color=(125, 123, 114))
    # new_img = PIL.Image.new('RGB', padded_size, color=(0, 0, 0))
    new_img.paste(img, (v, v))
    top = random.randint(0, v*2)
    left = random.randint(0, v*2)
    new_img = new_img.crop((left, top, left + img.size[0], top + img.size[1]))
    return new_img



def Cutout_default(img, v):  
    # Passed random number generation test
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    # x = np.random.uniform(w)
    # y = np.random.uniform(h)
    if v <= 16: # for cutout of cifar and SVHN
        assert w == h == 32
        x = random.uniform(0, w)
        y = random.uniform(0, h)

        x0 = int(min(w, max(0, x - v // 2))) # clip to the range (0, w)
        x1 = int(min(w, max(0, x + v // 2)))
        y0 = int(min(h, max(0, y - v // 2)))
        y1 = int(min(h, max(0, y + v // 2)))

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        # img = CutoutAbs(img, v)
        return img
    else:
        raise NotImplementedError

def RandCrop(img, _): # 0~4, max: 10
    v = 4
    return mean_pad_randcrop(img, v)

def RandCutout(img, _): # 0~16, max: 20
    v = 16  # Cutout 0.5 means 0.5*32=16 pixels as in the FastAA paper
    return Cutout_default(img, v)

# def RandCrop(img, _, aug_str=-1): # 0~4, max: 10
#     if aug_str >= 0:
#         v = round(4*aug_str)
#     else:
#         v = 4
#     return mean_pad_randcrop(img, v)

# def RandCutout(img, _, aug_str=-1): # 0~16, max: 20
#     if aug_str >= 0:
#         v = round(16*aug_str)
#     else:
#         v = 16  # Cutout 0.5 means 0.5*32=16 pixels as in the FastAA paper
#     return Cutout_default(img, v)

def RandCutout60(img, _):
    v = 60  # Cutout 0.5 means 0.5*32=16 pixels as in the FastAA paper
    return Cutout_default(img, v)

def RandFlip(img, _):
    if random.random() > 0.5:
        img = Flip(img, None)
    return img

def Identity(img, _):
    return img



def _parse_fill(fill, img, name="fillcolor"):
    # Process fill color for affine transforms
    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if isinstance(fill, (list, tuple)):
        if len(fill) != num_bands:
            msg = ("The number of elements in 'fill' does not match the number of "
                   "bands of the image ({} != {})")
            raise ValueError(msg.format(len(fill), num_bands))

        fill = tuple(fill)

    return {name: fill}


def pad(img, padding_ltrb, fill=0, padding_mode='constant'):
    if isinstance(padding_ltrb, list):
        padding_ltrb = tuple(padding_ltrb)
    if padding_mode == 'constant':
        opts = _parse_fill(fill, img, name='fill')
        if img.mode == 'P':
            palette = img.getpalette()
            image = PIL.ImageOps.expand(img, border=padding_ltrb, **opts)
            image.putpalette(palette)
            return image
        return PIL.ImageOps.expand(img, border=padding_ltrb, **opts)
    elif len(padding_ltrb) == 4:
        image_width, image_height = img.size
        cropping = -np.minimum(padding_ltrb, 0)
        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop((crop_left, crop_top, image_width - crop_right, image_height - crop_bottom))
        pad_left, pad_top, pad_right, pad_bottom = np.maximum(padding_ltrb, 0)

        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
            img = PIL.Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return PIL.Image.fromarray(img)
    else:
        raise Exception

def augment_list(for_autoaug=True, for_DeepAA_cifar=True, for_DeepAA_imagenet=True):  # 16 oeprations and their ranges
    # upper mag only for contrast, color, brightness, sharpness 
    # l = [
    #     (ShearX, -0.3, 0.3),  # 0
    #     (ShearY, -0.3, 0.3),  # 1
    #     (TranslateX, -0.45, 0.45),  # 2
    #     (TranslateY, -0.45, 0.45),  # 3
    #     (Rotate, -30, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5, param X
    #     (Invert, 0, 1),  # 6, param X
    #     (Equalize, 0, 1),  # 7, param x
    #     (Solarize, 0, 256),  # 8
    #     (Posterize, 4, 8),  # 9, !!!! '8' is original
    #     (Contrast, 1.0, 1.9),  # 10
    #     (Color, 1.0, 1.9),  # 11
    #     (Brightness, 1.0, 1.9),  # 12
    #     (Sharpness, 1.0, 1.9),  # 13
    #     (Cutout, 0, 0.2),  # 14, !!!! '0' is original
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]
    # both mag
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5, param X
        (Invert, 0, 1),  # 6, param X
        (Equalize, 0, 1),  # 7, param x
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9, !!!! '8' is original
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14, !!!! '0' is original
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    if for_DeepAA_cifar:
        l += [
            (Identity, 0., 1.0),
            (RandFlip, 0., 1.0), # Additional 15, px
            (RandCutout, 0., 1.0), # 16, px
            (RandCrop, 0., 1.0), # 17, px
        ]

    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def Cutout16(img, _):
    # return CutoutAbs(img, 16)
    return Cutout_default(img, 16)

def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

# def apply_augment(img, name, level):
#     augment_fn, low, high = get_augment(name)
#     if name in ['RandCutout','RandCrop']:
#         return augment_fn(img.copy(), -1, aug_str=level)
#     else:
#         return augment_fn(img.copy(), level * (high - low) + low)

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
