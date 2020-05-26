# -*- coding: utf-8 -*-
"""
Created on Mon April 20 18:05:56 2020

@author: Matsui
"""
from copy import deepcopy   # 深い複製
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import configparser

import chainer
import chainer.functions as F

from chainer import cuda
from chainer import optimizers
from chainer import serializers
from segnet_networks import SegNetBasic, SegNetBasicOnly, SegNetBasicAll

from loader import CamVid_loader


def load_CamVid():
    cp = configparser.ConfigParser()
    cp.read('config')
    root_dir = cp.get('dataset_dir', 'dir_path')
    x_train, x_test, c_train, c_test = CamVid_loader(root_dir)
    num_train = len(x_train)
    num_test = len(x_test)
    return x_train, x_test, c_train, c_test, num_train, num_test


def training_parameters():
    cp = configparser.ConfigParser()
    cp.read('config')
    use_device = int(cp.get('Hyper_parameters', 'gpu_on'))
    num_epochs = int(cp.get('Hyper_parameters', 'number_epochs'))
    batch_size = int(cp.get('Hyper_parameters', 'batch_size'))
    learning_rate = float(cp.get('Hyper_parameters', 'learning_rate'))
    dropout_mode = cp.get('Hyper_parameters', 'dropout_mode')

    return use_device, num_epochs, batch_size, learning_rate, dropout_mode


def train_part(model, optimizer, num_train, x_train, c_train, batch_size):

    epoch_losses = []               # エポック内の損失値
    epoch_accs = []                 # エポック内の認識率
    model.cleargrads()
    for i in tqdm(range(0, num_train, batch_size)):
        x_batch = xp.asarray(x_train[i:i+batch_size], dtype=xp.float32)
        ｃ_batch = xp.asarray(c_train[i:i+batch_size], dtype=xp.int32)
        with chainer.using_config('train', True):
            y_batch = model(x_batch)

            # 損失関数の計算
            loss = F.softmax_cross_entropy(y_batch, c_batch)
            accuracy = F.accuracy(y_batch, c_batch)       # 認識率

            loss.backward()                 # 重みの更新

        optimizer.update()
        model.cleargrads()              # 勾配のリセット

        epoch_losses.append(loss.data)
        epoch_accs.append(accuracy.data)

    epoch_loss = np.mean(cuda.to_cpu(xp.stack(epoch_losses)))   # エポックの平均損失
    epoch_acc = np.mean(cuda.to_cpu(xp.stack(epoch_accs)))     # エポックの平均認識率
    train_loss_log.append(epoch_loss)
    train_acc_log.append(epoch_acc)

    return train_loss_log, train_acc_log, epoch_loss, epoch_acc


def validation(model, num_test, x_test, c_test, batch_size):      # バリデーション
    losses = []
    accs = []
    for i in tqdm(range(0, num_test, batch_size)):
        x_batch = xp.asarray(x_test[i:i+batch_size], dtype=xp.float32)
        ｃ_batch = xp.asarray(c_test[i:i+batch_size], dtype=xp.int32)

        x_batch = chainer.Variable(x_batch)
        ｃ_batch = chainer.Variable(c_batch)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                y_batch = model(x_batch)

                # 損失関数の計算
                loss = F.softmax_cross_entropy(y_batch, c_batch)
                accuracy = F.accuracy(y_batch, c_batch)       # 認識率

        losses.append(loss.data)
        accs.append(accuracy.data)
    test_loss = np.mean(cuda.to_cpu(xp.stack(losses)))
    test_acc = np.mean(cuda.to_cpu(xp.stack(accs)))     # エポックの平均認識率
    test_loss_log.append(test_loss)
    test_acc_log.append(test_acc)

    return test_loss_log, test_acc_log, test_loss, test_acc


def check_result(x_test, best_model):
    # 答え合わせ
    n = 4    # 確認枚数
    K = 11  # クラス数
    x_batch = xp.asarray(x_test[:n])
    y_batch = best_model(x_batch)
    y_batch = cuda.to_cpu(y_batch.data)
    for i in range(n):

        # 入力画像
        plt.figure(figsize=(9, 3))
        plt.imshow(cuda.to_cpu(x_batch[i].transpose(1, 2, 0)))
        plt.show()
        # 出力画像
        plt.figure(figsize=(9, 3))
        plt.matshow(y_batch[i].argmax(0) / (K - 1), cmap=plt.cm.rainbow)
        plt.show()


def save_best_model(test_loss, test_acc,
                    model, best_model, best_test_acc,
                    best_val_loss, best_epoch):
    # 最小損失ならそのモデルを保持
    if test_loss < best_val_loss:
        best_model = deepcopy(model)
        best_val_loss = test_loss
        best_test_acc = test_acc
        best_epoch = epoch

    else:
        best_model = best_model
        best_val_loss = best_val_loss
        best_test_acc = best_test_acc
        best_epoch = best_epoch

    return best_model, best_val_loss, best_test_acc, best_epoch


def print_result_log(epoch, train_loss_log, test_loss_log,
                     train_acc_log, test_acc_log, epoch_loss, epoch_acc):

    # エポック数、認識率、損失値の表示
    print('{}: loss = {}, accuracy = {}'.format(epoch, epoch_loss, epoch_acc))
    # グラフの表示
    # lossの推移
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(train_loss_log, label='train loss')
    plt.plot(test_loss_log, label='test loss')
    plt.legend()
    plt.grid()

    # accuracyの推移
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(train_acc_log, label='train acc')
    plt.plot(test_acc_log, label='test acc')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def complete_logs(best_epoch, best_val_loss, best_test_acc, num_epochs,
                  batch_size, learning_rate, dropout_mode):
    print('[Hyper Parameters]')
    print('Best Epoch = {}'. format(best_epoch))
    print('min loss = {}'. format(best_val_loss))
    print('max accuracy = {}'. format(best_test_acc))
    print('Epochs = {}'. format(num_epochs))
    print('batch size = {}'. format(batch_size))
    print('learning rate = {}'. format(learning_rate))
    if dropout_mode == 'none':
        print('Dropout layer is not used.')
    elif dropout_mode == 'only':
        print('Dropout layer is used at just before last layer.')
    elif dropout_mode == 'all':
        print('Dropout layer is used in all layers.')


if __name__ == '__main__':

    (x_train, x_test, c_train, c_test,
     num_train, num_test) = load_CamVid()

    (gpu, num_epochs, batch_size,
     learning_rate, dropout_mode) = training_parameters()

    xp = cuda.cupy if gpu >= 0 else np
    if dropout_mode == 'none':
        model = SegNetBasic()
    elif dropout_mode == 'only':
        model = SegNetBasicOnly()
    elif dropout_mode == 'all':
        model = SegNetBasicAll()

    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    train_loss_log = []     # 訓練損失関数log
    train_acc_log = []      # 訓練認識率log
    test_loss_log = []      # テスト用損失関数log
    test_acc_log = []       # テスト用認識率log
    best_model = []
    best_epoch = []
    best_val_loss = np.inf  # 損失関数最小値保持値
    best_test_acc = np.zeros
    epoch_loss = []
    epoch_acc = []

    try:
        for epoch in range(num_epochs):
            (train_loss_log, train_acc_log,
             epoch_loss, epoch_acc) = train_part(model, optimizer, num_train,
                                                 x_train, c_train,
                                                 batch_size)

            (test_loss_log, test_acc_log,
             test_loss, test_acc) = validation(model, num_test,
                                               x_test, c_test,
                                               batch_size)

            (best_model, best_val_loss,
             best_test_acc, best_epoch) = save_best_model(test_loss, test_acc,
                                                          model, best_model,
                                                          best_test_acc,
                                                          best_val_loss,
                                                          best_epoch)

            print_result_log(epoch, train_loss_log, test_loss_log,
                             train_acc_log, test_acc_log,
                             epoch_loss, epoch_acc)

    except KeyboardInterrupt:
        print('Interrupted by Ctrl+c!')

    # best_modelの保存
    serializers.save_npz('my.SegNetBasic', best_model)
    serializers.save_npz('my.state', optimizer)

    check_result(x_test, best_model)

    complete_logs(best_epoch, best_val_loss, best_test_acc,
                  num_epochs, batch_size, learning_rate, dropout_mode)
