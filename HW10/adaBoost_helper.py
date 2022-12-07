'''Import Statements'''
import numpy as np
import cv2
import os
from tqdm import tqdm
import math


class ImageLoader:
    def __init__(self, dir: str):
        """
        :param dir: directory where test/train data reside
        """
        self.dir = dir
        self.train = None       # train images/labels [(img1, label1), (img2, label2), ..., (imgN, labelN)]
        self.test = None        # test images/labels [(img1, label1), (img2, label2), ..., (imgN, labelN)]
        self.train_pos = 0      # num of positive train images
        self.train_neg = 0      # num of negative train images
        self.test_pos = 0       # num of positive test images
        self.test_neg = 0       # num of negative test images

    def load_train(self):
        train_pos_dir = os.path.join(self.dir, "train/positive")
        train_neg_dir = os.path.join(self.dir, "train/negative")

        train_images_pos_name = [f for f in os.listdir(train_pos_dir) if f.endswith('.png')]
        train_images_neg_name = [f for f in os.listdir(train_neg_dir) if f.endswith('.png')]

        train_image_pos_labels = [1] * len(train_images_pos_name)
        train_image_neg_labels = [0] * len(train_images_neg_name)

        self.train_pos = len(train_images_pos_name)
        self.train_neg = len(train_images_neg_name)

        assert (len(train_images_pos_name) == len(train_image_pos_labels))
        assert (len(train_images_neg_name) == len(train_image_neg_labels))

        pos_image = [cv2.imread(os.path.join(train_pos_dir, i), 0) for i in train_images_pos_name]
        neg_image = [cv2.imread(os.path.join(train_neg_dir, i), 0) for i in train_images_neg_name]

        all_labels = train_image_pos_labels + train_image_neg_labels
        all_image = pos_image + neg_image
        assert (len(all_labels) == len(all_image))

        self.train = [(all_image[i], all_labels[i]) for i in range(len(all_image))]
        assert (len(self.train) == len(all_image))

    def load_test(self):
        test_pos_dir = os.path.join(self.dir, "test/positive")
        test_neg_dir = os.path.join(self.dir, "test/negative")

        test_images_pos_name = [f for f in os.listdir(test_pos_dir) if f.endswith('.png')]
        test_images_neg_name = [f for f in os.listdir(test_neg_dir) if f.endswith('.png')]

        test_image_pos_labels = [1] * len(test_images_pos_name)
        test_image_neg_labels = [0] * len(test_images_neg_name)

        assert (len(test_images_pos_name) == len(test_image_pos_labels))
        assert (len(test_images_neg_name) == len(test_image_neg_labels))

        self.test_pos = len(test_image_pos_labels)
        self.test_neg = len(test_images_neg_name)

        pos_image = [cv2.imread(os.path.join(test_pos_dir, i), 0) for i in test_images_pos_name]
        neg_image = [cv2.imread(os.path.join(test_neg_dir, i), 0) for i in test_images_neg_name]

        all_labels = test_image_pos_labels + test_image_neg_labels
        all_image = pos_image + neg_image
        assert (len(all_labels) == len(all_image))

        self.test = [(all_image[i], all_labels[i]) for i in range(len(all_image))]
        assert (len(self.test) == len(all_image))


class Filter:
    def __init__(self, x_coord, y_coord, width, height):
        """
        :param x_coord: (int) x-coordinate to evaluate filter
        :param y_coord: (int) y-coordinate to evaluate filter
        :param width: (int) width of filter
        :param height: (int) height of filter
        """
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.width = width
        self.height = height

    def eval_filter(self, img):
        """
        :param img: integral image
        :return: (float) evaluates filter on img
        """
        A = img[self.y_coord + self.height][self.x_coord + self.width]
        B = img[self.y_coord][self.x_coord]
        C = img[self.y_coord + self.height][self.x_coord]
        D = img[self.y_coord][self.x_coord + self.width]
        return A + B - C - D


class AdaBoost:
    def __init__(self, T=10):
        """
        :param T: (int) number of weak classifiers
        """
        self.T = T
        self.alphas = list()             # list of trust factors
        self.weak_classifiers = list()   # list of weak classifiers


    def train(self, train_data, num_pos, num_neg):
        """
        :param train_data: train images/labels [(img1, label1), (img2, label2), ..., (imgN, labelN)]
        :param num_pos: (int) num of positive train images
        :param num_neg: (int) num of negative train images
        """
        weights = np.zeros(len(train_data))
        train_data_integral = list()
        for x in range(len(train_data)):
            train_data_integral.append((integral_image(train_data[x][0]), train_data[x][1]))
            if train_data[x][1] == 1:
                weights[x] = 1.0 / (2 * num_pos)
            else:
                weights[x] = 1.0 / (2 * num_neg)

        features = self.construct_filter1(train_data_integral[0][0].shape)
        X, y = self.extract_feature(features, train_data_integral, False)

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak_classifier(X, y, features, weights)
            clf, error, accuracy = self.select_best_classifier(weak_classifiers, weights, train_data_integral)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0 / beta)
            self.alphas.append(alpha)
            self.weak_classifiers.append(clf)
            # print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))

    def construct_filter1(self, image_shape):
        height, width = image_shape
        features = []
        h = 1
        for w in range(1, width + 1, 5):
            i = 0
            while i + w < width:
                j = 0
                while j + h < height:
                    features.append(([Filter(i + w, j, w, h)], [Filter(i, j, w, h)])) if i + 2 * w < width else None
                    j += 1
                i += 1
        return np.array(features, dtype=object)

    def extract_feature(self, features, training_data, load):
        if load:
            return np.load("features.npz"), np.load("labels.npz")
        else:
            X = np.zeros((len(features), len(training_data)))
            y = np.array(list(map(lambda data: data[1], training_data)))
            i = 0
            for positive_regions, negative_regions in features:
                feature = lambda ii: sum([pos.eval_filter(ii) for pos in positive_regions]) - sum(
                    [neg.eval_filter(ii) for neg in negative_regions])
                X[i] = list(map(lambda data: feature(data[0]), training_data))
                i += 1
                np.savez("features.npz", X)
                np.savez("labels.npz", y)
            return X, y

    def train_weak_classifier(self, X, y, features, weights):
        total_pos = 0
        total_neg = 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w
        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            visited_pos, visited_neg, weight_pos, weight_neg = 0, 0, 0, 0
            min_error = float('inf')
            best_feature, best_threshold, best_polarity = None, None, None
            for w, f, label in applied_feature:
                error = min(weight_neg + total_pos - weight_pos, weight_pos + total_neg - weight_neg)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if visited_pos > visited_neg else -1
                if label == 1:
                    visited_pos += 1
                    weight_pos += w
                else:
                    visited_neg += 1
                    weight_neg += w
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def select_best_classifier(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def classify(self, image):
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.weak_classifiers):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0


class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
        :param positive_regions: positive regions in filter
        :param negative_regions: negative regions in filter
        :param threshold: decision threshold
        :param polarity: classification polarity
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        """
        :param x: image
        :return: labels in {0,1}
        """
        feature = lambda ii: sum([pos.eval_filter(ii) for pos in self.positive_regions]) - sum([neg.eval_filter(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0


class CascadeBoost:
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, training):
        pos, neg = [], []
        for ex in training:
            if ex[1] == 1:
                pos.append(ex)
            else:
                neg.append(ex)

        for feature_num in self.layers:
            if len(neg) == 0:
                print("Stopping early. FPR = 0")
                break
            clf = AdaBoost(T=feature_num)
            clf.train(pos + neg, len(pos), len(neg))
            self.clfs.append(clf)
            false_positives = []
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            neg = false_positives

    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image) == 0:
                return 0
        return 1


'''Adapted Directly From 
https://realpython.com/lessons/integral-images/'''
def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii
