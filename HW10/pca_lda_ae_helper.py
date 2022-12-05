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


'''def PCA(vec_train_image, vec_test_image, train_labels, test_labels)
Input: vec_train_image, vec_test_image (train/test image normalized)
       train_labels, test_labels (train/test labels)
Output: accuracy_list (list of accuracies depending on subspace dim)
Purpose: Given labeled train/test images, perform PCA on varying k values'''
def PCA(vec_train_image, vec_test_image, train_labels, test_labels):
    k_max = 30
    accuracy_rate = list()

    for k in range(1, k_max):
        _, _, u = np.linalg.svd(vec_train_image.T@vec_train_image)
        W = vec_train_image @ u.T

        W_k = W[:, :k]

        for i in range(W_k.shape[1]):
            W_k[:, i] = W_k[:, i] / np.linalg.norm(W_k[:, i])

        W_k_mean = np.mean(W_k, axis=1).reshape(-1,1)
        y_train = W_k.T @ (vec_train_image - W_k_mean)
        y_train = y_train.T

        y_test = W_k.T @ (vec_test_image - W_k_mean)
        y_test = y_test.T

        assert(y_train.shape == y_test.shape)

        y_train_list = [y_train[i, :] for i in range(y_train.shape[0])]
        y_test_list = [y_test[i, :] for i in range(y_test.shape[0])]
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(y_train_list, train_labels)
        pred = knn.predict(y_test_list)

        error = np.mean(pred != test_labels)
        accuracy_rate.append(1-error)
    return accuracy_rate


'''def LDA(vec_train_image, vec_test_image, train_labels, test_labels)
Input: vec_train_image, vec_test_image (train/test image normalized)
       train_labels, test_labels (train/test labels)
Output: accuracy_list (list of accuracies depending on subspace dim)
Purpose: Given labeled train/test images, perform LDA on varying k values'''
def LDA(vec_train_image, vec_test_image, train_labels, test_labels):
    k_max = 30
    accuracy_rate = list()

    for k in range(1, k_max):
        global_mean = np.mean(vec_train_image, axis=1)
        class_mean = np.zeros((vec_train_image.shape[0], np.unique(train_labels).size))

        train_labels_np = np.array(train_labels)
        for i in range(1, np.unique(train_labels).size):
            class_mean[:, i-1] = np.mean(vec_train_image[:, train_labels_np == i], axis=1)

        W = class_mean-global_mean.reshape(-1,1)
        W_k = W[:,:k]


        y_train = W_k.T @ (vec_train_image - global_mean.reshape(-1,1))
        y_test = W_k.T @ (vec_test_image - global_mean.reshape(-1,1))

        y_train = y_train.T
        y_test = y_test.T

        assert(y_train.shape == y_test.shape)

        y_train_list = [y_train[i, :] for i in range(y_train.shape[0])]
        y_test_list = [y_test[i, :] for i in range(y_test.shape[0])]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(y_train_list, train_labels)
        pred = knn.predict(y_test_list)

        error = np.mean(pred != test_labels)
        accuracy_rate.append(1-error)
    return accuracy_rate

'''i wrote this 100%'''
class KNN:
    def __init__(self, N_neigh=3) -> None:
        self.K = N_neigh

    def fit(self, X, Y):
        self.train_X = X
        self.train_Y = Y

    def predict(self, X):
        # X = np.array(X)
        pred = []
        for x in X:
            args = np.argsort(np.sum((self.train_X - x) ** 2, axis=1))
            # print(args[1:self.K+1])
            label = np.bincount(self.train_Y[args[:self.K]]).argmax()
            pred.append(label)
        return pred

    def score(self, X, Y):
        pred = self.predict(X)
        num_matches = sum(pred == Y)
        return num_matches / len(pred)