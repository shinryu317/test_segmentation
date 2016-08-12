# -*- coding: utf-8 -*-
'''
seg_evaluater.py
==============================
シーンラベリングで用いる評価モジュール。
'''
import logging

import numpy as np


def check(class_map, label_map):
    '''
    評価する際に満たしてほしい条件をチェックする。引っかかったら
    警告を吐き出す。

    Arguments
    ----------------------------------------
    class_maps : numpy.ndarray
        テスト画像をモデルに入力して得られた推定クラスマップ。
    label_maps : numpy.ndarray
        テスト画像に対応する教師クラスマップ。
    '''
    if not class_map.dtype == label_map.dtype == int:
        logging.warning('Warning: Not match type of the class map and label map.')
    if not class_map.shape == label_map.shape:
        logging.warning('Warning: Not match shape of the class map and label map.')
    if not class_map.ndim == label_map.ndim:
        logging.warning('Warning: Not match number of dimensions of the class map and label map.')

def calc_global_accuracy(class_map, label_map):
    '''
    ピクセル毎に正誤判定をして正解した数を accuracy とする評価方法。
    単純な代わりに背景で精度を稼いでしまう欠点がある。
    FCN の論文における pixel accuracy に相当する。

    Arguments
    ----------------------------------------
    class_map : numpy.ndarray
        テスト画像をモデルに入力して得られた推定クラスマップ。
    label_map : numpy.ndarray
        テスト画像に対応する教師クラスマップ。

    Return
    ----------------------------------------
    global_accuracy : float
        全画素の正答率を計算して得られた精度。値のレンジは[0, 1]。
    '''
    check(class_map, label_map)
    tp_map = class_map == label_map
    n_tp = np.sum(tp_map)
    return float(n_tp) / label_map.size

def calc_class_accuracy(class_map, label_map, n_labels):
    '''
    クラス毎に精度を計算し、その平均を accuracy とする評価方法。
    FCN の論文における mean accuracy に相当する。

    Arguments
    ----------------------------------------
    class_map : numpy.ndarray
        テスト画像をモデルに入力して得られた推定クラスマップ。
    label_map : numpy.ndarray
        テスト画像に対応する教師クラスマップ。

    Return
    ----------------------------------------
    class_accuracy : float
        クラス毎に正答率を計算して得られた精度。値のレンジは[0, 1]。
    '''
    check(class_map, label_map)
    each_class_accuracy = np.zeros(n_labels, dtype='float')
    tp_map = class_map == label_map
    for label in xrange(n_labels):
        n_tp = np.sum(tp_map[class_map == label])
        n_pos = np.sum(label_map == label)
        each_class_accuracy[label] = float(n_tp) / max(1.0, n_pos)
    return np.mean(each_class_accuracy)

def calc_mean_iou(class_map, label_map, n_labels):
    '''
    クラス毎に IoU(intersection over union) を計算してその平均を返す。
    FCN の論文における mean IU に相当する。

    Arguments
    ----------------------------------------
    class_map : numpy.ndarray
        テスト画像をモデルに入力して得られた推定クラスマップ。
    label_map : numpy.ndarray
        テスト画像に対応する教師クラスマップ。

    Return
    ----------------------------------------
    mean_iou : float
        クラス毎に IoU を計算して得られた精度。値のレンジは[0, 1]。
    '''
    check(class_map, label_map)
    each_class_iou = np.zeros(n_labels, dtype='float')
    tp_map = class_map == label_map
    for label in xrange(n_labels):
        n_tp = np.sum(tp_map[class_map == label])
        n_pos = np.sum(label_map == label)
        n_res = np.sum(class_map == label)
        each_class_iou[label] = float(n_tp) / max(1.0, (n_pos + n_res - n_tp))
    return np.mean(each_class_iou)

def calc_accuracies(class_maps, label_maps, n_labels):
    '''
    シーンラベリングで用いられる評価方法すべてで評価する。
    具体的には, global accuracy, class accuracy, mean IoU の3つ。

    Arguments
    ----------------------------------------
    class_maps : numpy.ndarray
        テスト画像をモデルに入力して得られた全ての推定クラスマップ。
    label_maps : numpy.ndarray
        テスト画像に対応する全ての教師クラスマップ。
    n_labels : int
        教師信号のラベル数。出力クラス数ともいう。

    Returns
    ----------------------------------------
    global_accuracy : float
        calc_global_accuracy() で求めた精度。
    class_accuracy : float
        calc_class_accuracy() で求めた精度。
    mean_iou : float
        計算した IoU のクラス毎の平均、のサンプル数の平均。
    '''
    global_accuracy = np.zeros(len(class_maps), dtype='float')
    class_accuracy = np.zeros(len(class_maps), dtype='float')
    mean_iou = np.zeros(len(class_maps), dtype='float')
    for i, (class_map, label_map) in enumerate(zip(class_maps, label_maps)):
        global_accuracy[i] = calc_global_accuracy(class_map, label_map)
        class_accuracy[i] = calc_class_accuracy(class_map, label_map, n_labels)
        mean_iou[i] = calc_mean_iou(class_map, label_map, n_labels)
    return np.mean(global_accuracy), np.mean(class_accuracy), np.mean(mean_iou)