***************************************************************************************
***************************************************************************************

Matlab demo code for UH-BDNN

If you use/adapt our code, please appropriately cite our ECCV 2016 paper:
"Learning to Hash with Binary Deep Neural Network", Thanh-Toan Do, Anh-Dzung Doan and Ngai-Man Cheung

This code is for academic purpose only. Not for commercial/industrial activities.

***************************************************************************************
***************************************************************************************


I. PREREQUISITES
=================

1. A working version of Matlab/mex.
2. A working version of the Yael library (http://yael.gforge.inria.fr)
3. A 3rd party implementation of L-BFGS (subdirectory minFunc included in http://ufldl.stanford.edu/wiki/resources/sparseae_exercise.zip)

We have included, compiled and tested all 3rd party libraries on MATLAB R2014a, OS Ubuntu 14.04 LTS 64-bit



II. DATASET
=================

Download dataset.zip [https://drive.google.com/file/d/1IfDgF-LPnk07TNTF1iqVPYUKzlSsmjdZ/view?usp=sharing](here), extract and place it in the directory of source code

The folder './dataset' contains the mat files used for this demo code
		+ cifar-10/cifar_gist_320.mat: This is GIST 320-D representation of CIFAR-10, it contains:
				- Xtest (10000x320): 10K query images
				- Xtrain(50000x320): 50K gallery images
				- Ytest (10000x1): semantic label of 10K query images (we do not use in unsupervised case)
				- Ytrain(50000x1): semantic label of 50K gallery images (we do not use in unsupervised case)
        + cifar-10/groundtruth.mat: groundtruth of CIFAR-10 based on Euclidean distance

		+ mnist/mnist_train.mat: This is 784-D gray-scale training images of MNIST, it contains:
				- Xtrain (784x60000): 60K gallery images
				- Ytrain (60000x1): semantic label of 60K gallery images (we do not use in unsupervised case)
		+ mnist/mnist_test.mat: This is 784-D gray-scale query images of MNIST, it contains:
				- Xtest (784x10000): 10K query images
				- Ytest (10000x1): semantic label of 10K query images (we do not use in unsupervised case)
        + mnist/groundtruth.mat: groundtruth of MNIST based on Euclidean distance

III. USAGE
=================



Run 'demo.m', it will visualize a comparison between our method with ITQ in mAP
Please test with MNIST dataset first. If the code works properly, you will get Fig 2(b) (with 2 curves: our UH-BDNN and ITQ) as our ECCV16 paper
It should take 3-4 hours to finish, we tested on workstation Intel Xeon(R) CPU E5-1620 v2 @ 3.70GHz × 8, RAM 64GB


IV. CORE FUNCTIONS:
=================
1. learn_all.m: learns binary deep neural network
		- INPUT:
				- Xtrain (number of training sample X dimension): training data
				- Xval (number of validation sample X dimension): validation data
				- val_gnd_inds (number of validation sample X number of grountruth): groundtruth indices for validation data. 
				- hiddenSize: array contains number of layers in all hidden layers. In our code, we fix number of hidden layers is equal to 3
				- lambda1, lambda2, lambda3, lambda4: lambda values respectively correspond to regularization, binary constraint violation, independence and balance
				- iter_lbfgs: number of L-BFGS iteration 
				- max_iter: maximum number of iteration for alternating optimization over (W,c) and B
		- OUTPUT:
				- stack: cell contains (W,c) in our binary deep neural network.
				
2. feedForwardDeep.m: does feedforward given an input sample, output is obtained from penultimate layer (layer n-1)
		- INPUT:
				- stack: cell contains (W,c) in our binary deep neural network (output of learn_all.m function)
				- data (dimension X number of sample): the input sample.
