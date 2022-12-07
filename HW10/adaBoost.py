import numpy as np
import cv2
import os
from tqdm import tqdm
import math

class ImageSet:
    def __init__(self, dir:str):
        self.dir = dir
        self.train = None
        self.test = None
        self.train_pos = 0
        self.train_neg = 0
        self.test_pos = 0
        self.test_neg = 0

    def load_train(self):
        train_pos_dir = os.path.join(self.dir, "train/positive")
        train_neg_dir = os.path.join(self.dir, "train/negative")

        train_images_pos_name = [f for f in os.listdir(train_pos_dir) if f.endswith('.png')]
        train_images_neg_name = [f for f in os.listdir(train_neg_dir) if f.endswith('.png')]

        train_image_pos_labels = [1] * len(train_images_pos_name)
        train_image_neg_labels = [0] * len(train_images_neg_name)

        self.train_pos = len(train_images_pos_name)
        self.train_neg = len(train_images_neg_name)

        assert(len(train_images_pos_name) == len(train_image_pos_labels))
        assert(len(train_images_neg_name) == len(train_image_neg_labels))

        pos_image = [cv2.imread(os.path.join(train_pos_dir, i), 0) for i in train_images_pos_name]
        neg_image = [cv2.imread(os.path.join(train_neg_dir, i), 0) for i in train_images_neg_name]

        all_labels = train_image_pos_labels + train_image_neg_labels
        all_image = pos_image + neg_image
        assert(len(all_labels) == len(all_image))

        self.train = [(all_image[i], all_labels[i]) for i in range(len(all_image))]
        assert(len(self.train) == len(all_image))

    def load_test(self):
        test_pos_dir = os.path.join(self.dir, "test/positive")
        test_neg_dir = os.path.join(self.dir, "test/negative")

        test_images_pos_name = [f for f in os.listdir(test_pos_dir) if f.endswith('.png')]
        test_images_neg_name = [f for f in os.listdir(test_neg_dir) if f.endswith('.png')]

        test_image_pos_labels = [1] * len(test_images_pos_name)
        test_image_neg_labels = [0] * len(test_images_neg_name)

        assert(len(test_images_pos_name) == len(test_image_pos_labels))
        assert(len(test_images_neg_name) == len(test_image_neg_labels))

        self.test_pos = len(test_image_pos_labels)
        self.test_neg = len(test_images_neg_name)

        pos_image = [cv2.imread(os.path.join(test_pos_dir, i), 0) for i in test_images_pos_name]
        neg_image = [cv2.imread(os.path.join(test_neg_dir, i), 0) for i in test_images_neg_name]

        all_labels = test_image_pos_labels + test_image_neg_labels
        all_image = pos_image + neg_image
        assert(len(all_labels) == len(all_image))

        self.test = [(all_image[i], all_labels[i]) for i in range(len(all_image))]
        assert(len(self.test) == len(all_image))


class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        return ii[self.y+self.height][self.x+self.width] \
               + ii[self.y][self.x] \
               - (ii[self.y+self.height][self.x] + ii[self.y][self.x+self.width])


class AdaBoost:
    def __init__(self, T=10):
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, train_data, num_pos, num_neg):
        weights = np.zeros(len(train_data))
        train_data_integral = list()
        for x in range(len(train_data)):
            train_data_integral.append((integral_image(train_data[x][0]), train_data[x][1]))
            if train_data[x][1] == 1:
                weights[x] = 1.0 / (2 * num_pos)
            else:
                weights[x] = 1.0 / (2 * num_neg)
        features = self.build_features(train_data_integral[0][0].shape)
        X, y = self.apply_features(features, train_data_integral, False)

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, train_data_integral)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0 / beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))

    def build_features(self, image_shape):
        height, width = image_shape
        features = []
        h = 1
        for w in range(1, width + 1, 1):
            # for h in range(1, height + 1):
            i = 0
            while i + w < width:
                j = 0
                while j + h < height:
                    # 2 rectangle features
                    immediate = RectangleRegion(i, j, w, h)
                    right = RectangleRegion(i + w, j, w, h)
                    if i + 2 * w < width:  # Horizontally Adjacent
                        features.append(([right], [immediate]))

                    # bottom = RectangleRegion(i, j + h, w, h)
                    # if j + 2 * h < height:  # Vertically Adjacent
                    #     features.append(([immediate], [bottom]))
                    #
                    # right_2 = RectangleRegion(i + 2 * w, j, w, h)
                    # # 3 rectangle features
                    # if i + 3 * w < width:  # Horizontally Adjacent
                    #     features.append(([right], [right_2, immediate]))
                    #
                    # bottom_2 = RectangleRegion(i, j + 2 * h, w, h)
                    # if j + 3 * h < height:  # Vertically Adjacent
                    #     features.append(([bottom], [bottom_2, immediate]))
                    #
                    # # 4 rectangle features
                    # bottom_right = RectangleRegion(i + w, j + h, w, h)
                    # if i + 2 * w < width and j + 2 * h < height:
                    #     features.append(([right, bottom], [immediate, bottom_right]))

                    j += 1
                i += 1
        return np.array(features, dtype=object)

    def apply_features(self, features, training_data, load):
        if load == True:
            return np.load("features.npz"), np.load("labels.npz")
        else:
            X = np.zeros((len(features), len(training_data)))
            y = np.array(list(map(lambda data: data[1], training_data)))
            i = 0
            for positive_regions, negative_regions in tqdm(features):
                feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum(
                    [neg.compute_feature(ii) for neg in negative_regions])
                X[i] = list(map(lambda data: feature(data[0]), training_data))
                i += 1
                np.savez("features.npz", X)
                np.savez("labels.npz", y)
            return X, y

    def train_weak(self, X, y, features, weights):
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w
        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1
                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def select_best(self, classifiers, weights, training_data):
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
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0


class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0


def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii