# -*- coding: utf-8 -*-
"""
Created on Mon April 20 13:35:35 2020

@author: Matsui
"""

import numpy as np
import matplotlib.pyplot as plt


from chainercv.datasets import CamVidDataset
from chainer.dataset import concat_examples


import configparser

cp = configparser.ConfigParser()
cp.read('config')
root_dir = cp.get('dataset_dir', 'dir_path')


def transform(in_data):
    img, label = in_data
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    return img, label


def CamVid_loader(dataset_dir=root_dir):

    # Dataset
    train = CamVidDataset(dataset_dir, split='train')
    test = CamVidDataset(dataset_dir, split='val')

    train = concat_examples(train)
    test = concat_examples(test)

    train_images = train[:][0]
    train_labels = train[:][1]
    test_images = test[:][0]
    test_labels = test[:][1]

    train_images /= 255.0
    test_images /= 255.0

    return train_images, test_images, train_labels, test_labels


if __name__ == '__main__':
    (train_images, test_images,
     train_labels, test_labels) = np.array(CamVid_loader())

    print(train_labels.shape)

    a = train_images[0].transpose(1, 2, 0)
    b = train_labels[0]
    c = test_images[0].transpose(1, 2, 0)
    d = test_labels[0]


    plt.imshow(a)
    plt.show()
    plt.imshow(b)
    plt.show()
    plt.imshow(c)
    plt.show()
    plt.imshow(d)
    plt.show()
