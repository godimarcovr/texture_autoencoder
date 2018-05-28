import os
import random
import glob
import copy

import numpy as np
import scipy.io
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as transF

mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)


class TextureDataset(Dataset):
    """Texture dataset loader class."""

    def __init__(self, scores_mat_path, root_dir, transform=None, split_inds=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scores_mat_path = scores_mat_path
        tmp_mat = scipy.io.loadmat(scores_mat_path)
        self.scores = tmp_mat["enc_features"]
        self.scores -= np.min(np.reshape(self.scores, (-1, )))
        self.scores /= np.max(np.reshape(self.scores, (-1,))) #[0, 1]
        self.labels = tmp_mat['labels'][0] - 1
        tmp_img_paths = glob.glob(os.path.join(self.root_dir, "*", "*.*"))
        self.imgs = []
        for i, p in enumerate(tmp_img_paths):
            self.imgs.append(np.asarray(Image.open(p)))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                RandomScalingTransform(),
                RandomRotation90Transform(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                self.__class__.PIL_to_torch_transform()
            ])
        if split_inds is not None:
            self.set_split_inds(split_inds)

    def set_split_inds(self, split_inds):
        self.split_inds = split_inds
        self.imgs = [self.imgs[i] for i in self.split_inds]
        self.labels = self.labels[self.split_inds]
        self.scores = self.scores[self.split_inds]

    def split_dataset(self, newratio):
        all_inds = [x for x in range(len(self.imgs))]
        random.shuffle(all_inds)
        n_old_inds = int((1.0 - newratio) * len(all_inds))
        n_new_inds = len(all_inds) - n_old_inds
        new_ds = copy.deepcopy(self)
        if not hasattr(self, 'split_inds'):
            self.set_split_inds(all_inds[:n_old_inds])
            tmp_split_inds2 = all_inds[n_old_inds:]
            new_ds.set_split_inds(tmp_split_inds2)
        else:
            # TODO UNTESTED!
            tmp_split_inds = [self.split_inds[i] for i in all_inds[:n_old_inds]]
            tmp_split_inds2 = [self.split_inds[i] for i in all_inds[n_old_inds:]]
            self.set_split_inds(all_inds[:n_old_inds])
            self.split_inds = tmp_split_inds
            new_ds.set_split_inds(all_inds[n_old_inds:])
            new_ds.split_inds = tmp_split_inds2
        new_ds.transform = TextureDataset.inference_transform()
        return new_ds

    def __len__(self):
        return self.scores.shape[0]

    def __getitem__(self, idx):
        # data, scores, orig
        return self.transform(Image.fromarray(self.imgs[idx])), self.scores[idx], self.imgs[idx], self.labels[idx]

    @staticmethod
    def PIL_to_torch_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mu, sd)
        ])

    @staticmethod
    def inference_transform():
        return transforms.Compose([
                RandomScalingTransform(scales=[1]),
                TextureDataset.PIL_to_torch_transform()
            ])

def variable_size_input_collate_fn(batch):
    data = [item[0] for item in batch]
    scores = [item[1] for item in batch]
    orig = [item[2] for item in batch]
    label = [item[3] for item in batch]
    return [data, scores, orig, label]

def from_tensor_to_numpy(tensorimg):
    if len(tensorimg.size()) == 4:
        tensorimg = tensorimg[0]
    # tmp = tensorimg.cpu() * torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(sd), 1), 1) \
    #                 + torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(mu), 1), 1)
    tmp = tensorimg.cpu()
    tmp = tmp.clamp(0, 1)
    # tmp = np.reshape(tmp.numpy(), (tmp.size(1), tmp.size(2), tmp.size(0)))
    tmp = np.asarray(transforms.ToPILImage()(tmp))
    return tmp

class RandomScalingTransform(object):
    def __init__(self, scales=[1/2, 1], power_of_two=3):
        self.scales = scales
        self.pow2 = 2 ** power_of_two

    def __call__(self, img):
        i = random.randrange(0, len(self.scales))
        # img.size ritorna (w,h) mentre il resize vuole (h, w)
        newsize = (int(img.size[1] * self.scales[i]), int(img.size[0] * self.scales[i]))

        newsize_smaller = ((newsize[0] // self.pow2) * self.pow2, (newsize[1] // self.pow2) * self.pow2)
        newsize_bigger = ((newsize[0] // self.pow2 + 1) * self.pow2, (newsize[1] // self.pow2 + 1) * self.pow2)

        area = newsize[0] * newsize[1]
        area_smaller = newsize_smaller[0] * newsize_smaller[1]
        area_bigger = newsize_bigger[0] * newsize_bigger[1]

        newsize = newsize_smaller if (area - area_smaller) < (area - area_bigger) else newsize_bigger
        return transF.resize(img, newsize, Image.BICUBIC)

class RandomRotation90Transform(object):
    def __call__(self, img):
        angle = random.randrange(0, 4) * 90
        # img.size ritorna (w,h) mentre il resize vuole (h, w)
        return transF.rotate(img, angle, resample=Image.BICUBIC, expand=True)

#TODO cuda, channel swap random transform, faster split, labels, random crops and zooms?

if __name__ == "__main__":
    train_ds = TextureDataset("ved47_features.mat", "dtd")
    val_ds = train_ds.split_dataset(0.1)
    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=variable_size_input_collate_fn, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, collate_fn=variable_size_input_collate_fn, shuffle=True)
    for batch_idx, (data, scores, orig, lab) in enumerate(val_dl):
        plt.subplot(1, 2, 1)
        plt.imshow(orig[0])
        plt.subplot(1, 2, 2)
        plt.imshow(from_tensor_to_numpy(data[0]))
        plt.draw()
        plt.waitforbuttonpress()