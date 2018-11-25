# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:10:56 2018

@author: Vishal Ghorpade
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:37:59 2018

@author: Vishal Ghorpade
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

def load_model():
    vgg =models.vgg(pretrained=True).features

    for param in vgg.parameters():
        param.requires_grad_(False)
   
    return vgg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg.to(device)


def load_image(path):
    image=Image.open(path).convert('rgb)
   
    transforms=transforms.Compose([transforms.resize(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image


def im_convert(tensor):
    """ Display a tensor as an image. """
   
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
   
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '' : 'conv2_1',
                  '' : 'conv3_1',
                  '' : 'conv4_1',
                  '' : 'conv4_2',
                  '' : 'conv5_1'}
       
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
           
    return features

def gram_matrix(tensor):
   
    _,d,h,w=tensor.size()
   
    tensor=tensor.view(d,h*w)
    gram=torch.mm(tenosr,tensor.t())
   
    return gram

def main():
    content_image=load_image(r"")
    style_image=load_image(r"")
    vgg=load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)
    print(vgg)
   
   
    content_feature=get_features(content_image,vgg)
    style_feature=get_features(style_image,vgg)
   
    style_grams={layer:gram_matrix(style_feature[layer] for layer in style_feature}
   
    target = content.clone().requires_grad_(True).to(device)
    style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta
   

    show_every = 400

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = 2000  # decide how many iterations to update your image (5000)

    for ii in range(1, steps+1):
        target_features = get_features(target,vgg)
        content_loss = torch.mean((target_features['conv4_2']-content_feature['conv4_2'])**2)
   
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
           target_feature = target_features[layer]
           _, d, h, w = target_feature.shape
           target_gram = gram_matrix(target_feature)
           style_gram =style_grams[layer]
           layer_style_loss = style_weights[layer]*torch.mean((target_gram-style_gram)**2)
       
           # add to the style loss
           style_loss += layer_style_loss / (d * h * w)
       
       
        total_loss = style_loss+content_loss
   
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
   
        # display intermediate images and print the loss
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.show()