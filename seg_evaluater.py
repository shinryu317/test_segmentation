# -*- coding: utf-8 -*-
'''
seg_evaluater.py
==============================
Evaluation module for semantic segmentation task.
'''
import numpy as np


def _verify(class_map, label_map, n_classes):
    assert class_map.ndim == label_map.ndim == 2  # height and width
    assert class_map.shape == label_map.shape
    assert np.max(class_map) < n_classes
    assert isinstance(class_map, np.ndarray)
    assert isinstance(label_map, np.ndarray)
    assert isinstance(n_classes, int)

def _generate_conf_matrix(class_map, label_map, n_classes=2):
    predicts = class_map.ravel().tolist()
    labels = label_map.ravel().tolist()
    conf_matrix = [[0] * n_classes for i in range(n_classes)]
    for predict, label in zip(predicts, labels):
        conf_matrix[predict][label] += 1
    return np.asarray(conf_matrix, dtype=np.uint16)

def calc_accuracies(class_map, label_map, n_classes=2):
    '''
    Calculate 3 metrics (global accuracy, class accuracy and mean IoU).
    '''
    _verify(class_map, label_map, n_classes)
    conf_matrix = _generate_conf_matrix(class_map, label_map, n_classes)
    tp = np.diag(conf_matrix).astype('float')
    res = np.sum(conf_matrix, axis=0).astype('float')
    pos = np.sum(conf_matrix, axis=1).astype('float')

    global_accuracy = np.sum(tp) / np.sum(conf_matrix)
    class_accuracy = np.mean(tp / np.maximum(1.0, pos))
    mean_iou = np.mean(tp / np.maximum(1.0, res + pos - tp))
    return np.asarray((global_accuracy, class_accuracy, mean_iou))
