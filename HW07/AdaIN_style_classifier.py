#!/usr/bin/env python
# coding: utf-8

# # Adaptive Instance Normalization (adaIN) Style Classifier

# ### Import Statements

# In[1]:


from style_classifier_helper import *
from tqdm import tqdm
import random
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay


# ### Load Training and Testing Image Data
# * train_img_list is a list of all the training images stored as np.ndarry
# * train_label_list is a list of the labels for the training images
# * test_img_list is a list of all the testing images stored as np.ndarry
# * test_label_list is a list of the labels for the test images

# In[2]:


training_directory = "/home/jo_wang/Desktop/ECE661/HW07/data/training"
test_directory = "/home/jo_wang/Desktop/ECE661/HW07/data/testing"

train_img_list, train_label_list = loadImages(training_directory)
test_img_list, test_label_list = loadImages(test_directory)

assert(len(train_img_list) == len(train_label_list))
assert(len(test_img_list) == len(test_label_list))
assert(len(train_img_list) == 920)
assert(len(test_img_list) == 200)


# ### Obtain Feature Maps of all Training Images
# 1. Create an instance of the VGG19 class
# 2. Load the pre-trained weights
# 3. Iterate across both the test and train data
# 4. Extract feature map from the CNN
# 5. Compute the gram matrix for each image and store in the respective list
# 6. Display gram matrix plots for one image in each class

# In[4]:


# Load the model and the provided pretrained weights
vgg = VGG19()
vgg.load_weights('/home/jo_wang/Desktop/ECE661/HW07/vgg_normalized.pth')

train_extracted_feature = list()
for i in tqdm(range(len(train_img_list))):
    ft = vgg(train_img_list[i])
    ft = np.resize(ft, (512, 256))
    train_extracted_feature.append(ft)

test_extracted_feature = list()
for i in tqdm(range(len(test_img_list))):
    ft = vgg(test_img_list[i])
    ft = np.resize(ft, (512, 256))
    test_extracted_feature.append(ft)

assert(len(train_extracted_feature) == len(train_img_list))
assert(len(test_extracted_feature) == len(test_img_list))


# ### Peform Adaptive Instance Normalization
# 1). Compute the mean and standard deviation of each row in the extracted feature map
# 2). Build a (920 x 1024) train and a (200 x 1024) test feature vector
# 3). Fit an SVM model with the train feature vector
# 4). Evaluate the SVM on the test feature vector
# 5). Compute accuracy and display the confusion matrix

# In[45]:


train_features = np.zeros((1,1024))
test_features = np.zeros((1,1024))

for ft in train_extracted_feature:
    mean = np.mean(ft, axis=1)
    std = np.std(ft, axis=1)
    temp = np.hstack((mean, std))
    temp = np.reshape(temp, (1,1024))
    train_features = np.vstack((train_features, temp))

for ft in test_extracted_feature:
    mean = np.mean(ft, axis=1)
    std = np.std(ft, axis=1)
    temp = np.hstack((mean, std))
    temp = np.reshape(temp, (1,1024))
    test_features = np.vstack((test_features, temp))

assert(train_features[1:,:].shape == (920, 1024))
assert(test_features[1:,:].shape == (200, 1024))

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train_features[1:,:], train_label_list)
texture_predict = clf.predict(test_features[1:,:])
print("Accuracy:",metrics.accuracy_score(test_label_list, texture_predict))
ConfusionMatrixDisplay.from_estimator(clf, test_features[1:,:], test_label_list)

