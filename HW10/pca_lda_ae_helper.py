import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

'''def load_train_data(dir):
Input:  dir (str) path to face train/test data
Output: train_images (list)
        train_labels (list)
        test_images  (list)
        test_labels  (list)
Purpose: Given the directory, load test/train data'''
def load_face_data(dir:str):
    train_path = os.path.join(dir, "train")
    test_path = os.path.join(dir, "test")

    train_images_names = [f for f in os.listdir(train_path) if f.endswith('.png')]
    train_labels = [int(f.split('_')[0]) for f in train_images_names]
    train_images = [cv2.imread(os.path.join(train_path, f), 0) for f in train_images_names]

    test_images_names = [f for f in os.listdir(test_path) if f.endswith('.png')]
    test_labels = [int(f.split('_')[0]) for f in test_images_names]
    test_images = [cv2.imread(os.path.join(test_path, f), 0) for f in test_images_names]
    return train_images, train_labels, test_images, test_labels

