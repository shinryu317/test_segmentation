# -*- coding: utf-8 -*-
'''
test_segmentation.py
==================================================
caffe を対象としたシーンラベリングの評価ツール。
global accuracy, class accuracy, mean IoU の評価ができる。
'''
from __future__ import print_function
import argparse
import imp
import logging
import os
from os import path

import caffe
import cv2
import numpy as np

import seg_evaluater


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        help='Pairs of image and label sample definition text file.',
    )
    parser.add_argument(
        '--basepath',
        nargs=2, default=('.', '.'),
        help='Path to a image and label samples.',
    )
    parser.add_argument(
        '--palette',
        required=True,
        help='Color to paint in a class map.',
    )
    parser.add_argument(
        '--save_folder',
        default='.',
        help='Output directory of a painted class map.',
    )
    parser.add_argument(
        '--prototxt',
        required=True,
        help='Model definition file.',
    )
    parser.add_argument(
        '--caffemodel',
        required=True,
        help='Trained model\'s parameter file.',
    )
    parser.add_argument(
        '--gpu_id',
        type=int, default=-1,
        help='Using gpu number.',
    )
    return parser.parse_args()

def start_logging():
    '''
    ロギングを開始する。
    表示するログは info 以上。
    '''
    date_format = '%m/%d %H:%M:%S'
    log_format = '[%(asctime)s %(module)s.py:%(lineno)d] '
    log_format += '%(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format
    )

def load_module(module_path):
    '''
    モジュールを読み込む。ここで読み込むのはクラスマップに色付けするパレット。
    つまり、クラスマップに塗る色の情報。

    Argument
    ------------------------------
    module_path : str
        モジュールへのパス(フォルダ・拡張子含む)

    Return
    ------------------------------
    palette : numpy.ndarray
        読み込んだモジュールに書かれた色のパレット。
    '''
    if not path.isfile(module_path):
        raise ImportError('Not found a module: {}'.format(module_path))
    head, tail = path.split(module_path)
    module_name = path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    module = imp.load_module(module_name, *info)
    return np.array(module.palette, dtype='uint8')

def load_samples(dataset, basepath):
    '''
    テキストファイルから1行ずつ読み込んで入力と教師に分離する。入力画像は
    カラー画像として読み込み，教師画像はグレースケール画像として読み込む。

    Arguments
    ------------------------------
    dataset : str
        画像ファイルと教師ファイルのペアが記述されているテキストファイル。
    basepath : list[str, str]
        dataset から読み込んだファイルのパス。ファイル名は含まない。
        1つ目は画像ファイル、2つ目は教師ファイルのパスに対応する。

    Returns
    ------------------------------
    inputs, labels : numpy.ndarray, numpy.ndarray
        入力サンプルと教師サンプル。
    '''
    try:
        with open(dataset, 'r') as f:
            files = [line.replace(',', '').split() for line in f]
            files = [[path.join(basepath[0], i), path.join(basepath[1], j)] for i, j in files]
            input_files, label_files = zip(*files)
            inputs = [cv2.imread(x, cv2.CV_LOAD_IMAGE_COLOR) for x in input_files]
            labels = [cv2.imread(y, cv2.CV_LOAD_IMAGE_GRAYSCALE) for y in label_files]
    except:
        raise IOError('Failed to load a test samples.')
    return np.asarray(inputs, dtype='uint8'), np.asarray(labels, dtype='int')

def evaluation(net, inputs):
    '''
    ネットワークモデルにテスト画像を入力してシーンラベリングをする。
    1 度にすべての画像を評価するとメモリが足りなくなる可能性があるため
    注意。batchsize は prototxt の値を使う。

    Arguments
    ------------------------------
    net : caffe._caffe.Net
        caffe.Net() で生成されたネットワーク。
    inputs : numpy.ndarray
        net に入れるテストサンプル。

    Return
    ------------------------------
    class_maps : numpy.ndarray
        テストサンプル数分のマップが格納されているクラスマップ。
    '''
    class_maps = np.ndarray(inputs.shape[:3], dtype=int)
    # [n_samples, height, width, channels] => [n_samples, channels, height, width]
    inputs = inputs.transpose(0, 3, 1, 2)
    if not net.blobs[net.inputs[0]].data[0].shape == inputs[0].shape:
        logging.warning('Warning: Not match shape of the blob and input.')

    batchsize = net.blobs[net.inputs[0]].data.shape[0]
    if not len(inputs) % batchsize == 0:
        logging.warning('Warning: The number of samples is not divisible by batch size. \
                         Can not be evaluated sample occurs.')
    for i in xrange(0, len(inputs), batchsize):
        x_batch = inputs[i:i + batchsize]
        probs = net.forward_all(**{net.inputs[0]:x_batch})
        class_maps[i] = probs[net.outputs[0]].argmax(axis=1)
    return class_maps

def visualization(save_folder, class_maps, palette):
    '''
    evaluation() で得られたクラスマップに対して palette を基に色を塗る。
    ついでに色を塗ったマップを save_folder に描き出す。save_folder で
    指定されたフォルダがない場合は自動的に作成する。

    Arguments
    ------------------------------
    save_folder : str
        クラスマップに色付けしたマップを描き出すフォルダ。
        ファイル名は連番になるのでパスだけ指定する。
    class_maps : numpy.ndarray
        色塗りするクラスマップ。入力サンプル数分のクラスマップが
        格納されている。
    palette : numpy.ndarray
        クラス毎に色を指定したBGRの配列。
    '''
    n_samples, map_height, map_width = class_maps.shape
    canvas = np.ndarray((n_samples, map_height, map_width, 3), dtype='uint8')
    for label in xrange(len(palette)):
        canvas[class_maps == label] = palette[label]
    if not path.exists(save_folder):
        os.makedirs(save_folder)
    for i, color_map in enumerate(canvas):
        save_file = path.join(save_folder, '{:05d}.jpg'.format(i))
        cv2.imwrite(save_file, color_map)

def calc_accuracy(class_maps, label_maps, n_labels):
    '''
    シーンラベリングの定量的評価を行う。
    評価方法は global accuracy, class accuracy, mean IoU の3つ。

    Arguments
    ------------------------------
    class_maps : numpy.ndarray
        evaluation() で得られたクラスマップ。int 型であれば自分で用意した
        クラスマップも評価できる。
    label_maps : numpy.ndarray
        入力サンプルに対応した教師サンプル。各画素がクラス番号になっている。
    '''
    global_acc, class_acc, mean_iou = \
        seg_evaluater.calc_accuracies(class_maps, label_maps, n_labels)
    logging.info('Global accuracy = {:0.4f} [%]'.format(100 * global_acc))
    logging.info('Class accuracy = {:0.4f} [%]'.format(100 * class_acc))
    logging.info('Mean IoU = {:0.4f} [%]'.format(100 * mean_iou))

def main():
    args = parse_arguments()
    start_logging()
    palette = load_module(args.palette)

    if not path.isfile(args.prototxt):
        raise IOError('Not found a prototxt: {}'.format(args.prototxt))
    if not path.isfile(args.caffemodel):
        raise IOError('Not found a caffemodel: {}'.format(args.caffemodel))

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    if 0 <= args.gpu_id:
        logging.info('Gpu mode. Using gpu device id: {}.'.format(args.gpu_id))
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
    else:
        logging.info('Cpu mode.')
        caffe.set_mode_cpu()

    inputs, labels = load_samples(args.dataset, args.basepath)
    logging.info('Loading dataset: {}'.format(args.dataset))
    logging.info('Loading prototxt: {}'.format(args.prototxt))
    logging.info('Loading caffemodel: {}'.format(args.caffemodel))
    logging.info('# of loaded samples: {}'.format(len(inputs)))

    logging.info('Evaluating...')
    class_maps = evaluation(net, inputs)
    visualization(args.save_folder, class_maps, palette)

    n_labels = net.blobs[net.outputs[0]].data.shape[1]
    calc_accuracy(class_maps, labels, n_labels)

if __name__ == '__main__':
    main()