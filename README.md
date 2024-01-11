[# Introduction to Machine Learning

# Description

This project contains several assignments related to the field of Machine Learning. Each assignment focuses on a different aspect of Machine Learning, providing practical experience with various algorithms and techniques.

# Assignments

## Assignment 1

**Description**: Using Logistic Regression and KNN wo classify particles as noise or signals based on a few attributes. 

**Approach**:
- `Preprocessing`: the dataset is not balanced due to technical reasons, so we need to balance it before training the models to avoid biases. So undersampling was used to resample the more dominant class. Also, the features had wildly different ranges to all features were normalized to assure they all converge easily.
- `Training`: Logistic Regression and KNN (with diffrent K values) were used to solve this problem and their performances where compared. The models were trained using 10-fold cross validation and the best model was selected based on the average accuracy of the folds.

## Assignment 2

### CIFAR-10_Image_classification
Using PyTorch to train Artificial neural network-based image classifier. The dataset used is CIFAR-10

This repository contains a report and an IPython Notebook for building and evaluating Artificial Neural Networks (ANN) and using transfer learning with pre-trained models for image classification.


**Description**: Fully Connected NN was used to classify images of 10 different classes. The dataset used was CIFAR-10. Then the results were compared to other pretrained models after we fine tuned them on our dataset.

**Approach**:
- `Preprocessing`: The images were normalized and resized to minimize training time and improve performance.  
- `Aproach`: Three fully connected NN were built and trained. The first one was a simple NN with 3 hidden layers of sizes 4096, 2048, 256. The second one was a deeper NN with 4 hidden layers of sizes 4096, 2048, 1024, 512. The third one was a deeper NN with 2 hidden layers of sizes 4096, 512. The models were trained using 10-fold cross validation and the best model was selected based on the average accuracy of the folds. Then transfer learning was used to fine tune pretrained models (VGG16, ResNet18, AlexNet) on our dataset and the results were compared to the results of the fully connected NNs.

### Assignment 3

**Description**: Using PCA to reduce the dimensionality of the dataset and then using KNN to classify the images. The dataset used was AT&T Faces dataset. the problem we are solving is face recognition and detection

**Approach**:

PCA algorithm was coded manually from scratch and applied on the Dataset with a different alphas. Then KNN was used to classify the images. The models were trained using 10-fold cross validation and the best model was selected based on the average accuracy of the folds. the performance was compared between different alphas and different K values. then the models were compared to the built in PCA functions in sklearn to assure the correctness of the implementation. Then images from the CIFAR-10 dataset were added ad the non-face images and KNN was used again this time to detect weather there was a face or not in the image.


### Assignment 4

**Description**: Image segmentation using K-means clustering algorithm. The dataset used was Berkeley Segmentation Benchmark (BSDS500). 

**Approach**:

k-means algorithm was used to segment the images with different K values. The results were compared to several ground truth segmentation in the dataset F-measure and Conditional Entropy were the metrics used to compare the results. then the segmentations were compared to the results of applying Normalized-cut algorithm on the images.

