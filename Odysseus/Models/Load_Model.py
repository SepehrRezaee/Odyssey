import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import copy
import pickle
import numpy as np
import glob
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from os import path
import math
from .mnist_architectures import Model_Google_1,Model_Google_2,Model_Google_3,Model_Google_4
from .FMNIST_architectures.vgg import VGG as FMVGG
from .FMNIST_architectures.resnet import ResNet18 as FMResNet18
from .FMNIST_architectures.lenet import LeNet as FMLeNet
from .FMNIST_architectures.preact_resnet import PreActResNet18 as FMPreActResNet18
from .FMNIST_architectures.googlenet import GoogLeNet as FMGoogLeNet
from .FMNIST_architectures.densenet import DenseNet121 as FMDenseNet121
from .Cifar10_models import *

# model loader for mnist models
def load_mnist_model(model_path, device,num_class=10, log=False):
    checkpoint = torch.load(model_path, map_location=device)
    if log:
        print("keys are :", checkpoint.keys())

    model = checkpoint['Architecture_Name']

    # Get the model
    if log:
        print('==> Building model..')
    if model == 'Model_Google_1':
        net = Model_Google_1()
    elif model == 'Model_Google_2':
        net = Model_Google_2()
    elif model == 'Model_Google_3':
        net = Model_Google_3()
    elif model == 'Model_Google_4':
        net = Model_Google_4()
    
    net = net.to(device)
    net.load_state_dict(checkpoint['net'])

    if 'test_clean_acc' in checkpoint:
        best_acc_clean = checkpoint['test_clean_acc']
        if log:
            print("The Accuracies on clean samples:  ", best_acc_clean)
    if 'test_trigerred_acc' in checkpoint:
        best_acc_trig = checkpoint['test_trigerred_acc']
        if log:
            print("The fooling rate: ", best_acc_trig)
    if 'Mapping' in checkpoint:
        mapping = checkpoint['Mapping']
        if log:
            print("Mapping is : ",mapping, type(mapping))
        if isinstance(mapping,int) or isinstance(mapping,np.float64):
            mapping=mapping*np.ones(num_class,dtype=float)
        elif isinstance(mapping,str):
            if mapping =='N/A':
                mapping = None
    else:
        mapping = None
    return net, mapping  # checkpoint['Mapping']

def load_model_fmnist(model_path, device, num_class=10):

    print('model path ',model_path)
    checkpoint = torch.load(model_path, map_location=device)
    print("keys are :", checkpoint.keys())
   
    model = checkpoint['Architecture_Name']

    print('==> Building model..')
    if model == 'Vgg19':
        net = FMVGG('VGG19')
    elif model == 'Resnet18':
        net = FMResNet18()
    elif model=='LeNet':
        net = FMLeNet()
    elif model == 'PreActResNet18':
        net = FMPreActResNet18()
    elif model == 'GoogleNet':
        net = FMGoogLeNet()
    elif model == 'DenseNet':
        net = FMDenseNet121()
  
    net = net.to(device)
    

    net.load_state_dict(checkpoint['net'])

    if 'test_clean_acc' in checkpoint:
        best_acc_clean = checkpoint['test_clean_acc']
        print("The Accuracies on clean samples:  ", best_acc_clean)
    if 'test_trigerred_acc' in checkpoint:
        best_acc_trig = checkpoint['test_trigerred_acc']
        print("The fooling rate: ", best_acc_trig)
    if 'Mapping' in checkpoint:
        mapping = checkpoint['Mapping']
        print("Mapping is : ",mapping)
        if isinstance(mapping,int):
            mapping=mapping*np.ones(num_class,dtype=float)
        elif isinstance(mapping,str):
            if mapping =='N/A':
                mapping = None
    else:
        mapping = None
    return net, mapping  # checkpoint['Mapping']


# model loader for Fashion_MNIST and Cifar10 models
def load_model_cifar10(model_path, device, num_class=10):
    print('model path ',model_path)
    checkpoint = torch.load(model_path, map_location=device)
    print("keys are :", checkpoint.keys())
   
    model = checkpoint['Architecture_Name']

    # Get the model
    print('==> Building model..')
    if model == 'Vgg19':
        net = VGG('VGG19')
    elif model == 'Resnet18':
        net = ResNet18()
    elif model == 'PreActResNet18':
        net = PreActResNet18()
    elif model == 'GoogleNet':
        net = GoogLeNet()
    elif model == 'DenseNet':
        net = DenseNet121()
    elif model == 'MobileNet':
        net = MobileNet()
    elif model == 'DPN92':
        net = DPN92()
    elif model == 'ShuffleNet':
        net = ShuffleNetG2()
    elif model == 'SENet':
        net = SENet18()
    elif model == 'EfficientNet':
        net = EfficientNetB0()
    else:
        net = MobileNetV2()

    net = net.to(device)
    

    net.load_state_dict(checkpoint['net'])

    if 'test_clean_acc' in checkpoint:
        best_acc_clean = checkpoint['test_clean_acc']
        print("The Accuracies on clean samples:  ", best_acc_clean)
    if 'test_trigerred_acc' in checkpoint:
        best_acc_trig = checkpoint['test_trigerred_acc']
        print("The fooling rate: ", best_acc_trig)
    if 'Mapping' in checkpoint:
        mapping = checkpoint['Mapping']
        print("Mapping is : ",mapping)
        if isinstance(mapping,int):
            mapping=mapping*np.ones(num_class,dtype=float)
        elif isinstance(mapping,str):
            if mapping =='N/A':
                mapping = None
    else:
        mapping = None
    return net, mapping  # checkpoint['Mapping']

