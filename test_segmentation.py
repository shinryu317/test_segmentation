# -*- coding: utf-8 -*-
'''
test_segmentation.py
==================================================
Evaluation tool for semantic segmentation task.
'''
from __future__ import print_function
import argparse
import imp
import logging
import os

import caffe
import cv2
import numpy as np

import seg_evaluater


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--testset', required=True,
        help='Pairs of image and label sample definition text file.'
    )
    parser.add_argument(
        '--prototxt', required=True,
        help='Model definition file.'
    )
    parser.add_argument(
        '--caffemodel', required=True,
        help='Trained model parameters.'
    )
    parser.add_argument(
        '--palette', required=True,
        help='Color to paint in a class map.'
    )
    parser.add_argument(
        '--basepath', nargs='*', default=('.', '.'),
        help='Path to a image and label samples.'
    )
    parser.add_argument(
        '--save_dir', default='.',
        help='Output directory of a painted class map.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=-1,
        help='Using gpu number.'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize flag of class maps.'
    )
    return parser.parse_args()

def start_logging():
    date_format = '%m/%d %H:%M:%S'
    log_format = '[%(asctime)s %(module)s.py:%(lineno)d] '
    log_format += '%(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format
    )

def load_module(module_path):
    if not os.path.isfile(module_path):
        raise IOError('Not found a palette: {}'.format(module_path))
    head, tail = os.path.split(module_path)
    module_name = os.path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    module = imp.load_module(module_name, *info)
    return np.asarray(module.palette, dtype='uint8')

def parse_text(testset, basepath='.'):
    data_list = []
    basepath = [basepath, basepath] if len(basepath) == 1 else basepath[:2]
    with open(testset, 'r') as f:
        for line in f:
            i, j = line.replace(',', ' ').split()
            data  = os.path.join(basepath[0], i)
            label = os.path.join(basepath[1], j)
            data_list.append((data, label))
    return data_list

def feedforward(net, filename):
    x = cv2.imread(filename, cv2.IMREAD_COLOR).transpose(2, 0, 1)
    x = x[np.newaxis, :, :, :]  # append minibatch
    prob = net.forward_all(**{net.inputs[0]:x})
    return prob[net.outputs[0]].argmax(axis=1)[0]

def visualize(class_map, palette, save_file):
    assert class_map.ndim == 2
    map_height, map_width = class_map.shape
    canvas = np.ndarray((map_height, map_width, 3), dtype='uint8')
    for label in range(len(palette)):
        canvas[class_map == label] = palette[label]
    cv2.imwrite(save_file, canvas)

def main():
    args = parse_arguments()
    start_logging()

    if not os.path.isfile(args.prototxt):
        raise IOError('Not found a prototxt: {}'.format(args.prototxt))
    if not os.path.isfile(args.caffemodel):
        raise IOError('Not found a caffemodel: {}'.format(args.caffemodel))

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    if 0 <= args.gpu_id:
        logging.info('Gpu mode. Using gpu device id: {}.'.format(args.gpu_id))
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
    else:
        logging.info('Cpu mode.')
        caffe.set_mode_cpu()

    palette = load_module(args.palette)
    testset = parse_text(args.testset, args.basepath)
    logging.info('testset: {}'.format(args.testset))
    logging.info('prototxt: {}'.format(args.prototxt))
    logging.info('caffemodel: {}'.format(args.caffemodel))
    logging.info('# of test samples: {}'.format(len(testset)))

    logging.info('Evaluating...')
    if args.visualize and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    n_classes = net.blobs[net.outputs[0]].data.shape[1]
    accuracies = np.array((0.0, 0.0, 0.0), dtype='float')
    for i, (x_file, y_file) in enumerate(testset):
        class_map = feedforward(net, x_file)
        label_map = cv2.imread(y_file, cv2.IMREAD_GRAYSCALE)
        accuracies += seg_evaluater.calc_accuracies(class_map, label_map, n_classes)
        if args.visualize:
            save_file = os.path.join(args.save_dir, '{:05d}.jpg'.format(i))
            visualize(class_map, palette, save_file)
    logging.info('Global accuracy = {:0.4f} [%]'.format(100. * accuracies[0] / len(testset)))
    logging.info('Class accuracy = {:0.4f} [%]'.format(100. * accuracies[1] / len(testset)))
    logging.info('Mean IoU = {:0.4f} [%]'.format(100. * accuracies[2] / len(testset)))

if __name__ == '__main__':
    main()
