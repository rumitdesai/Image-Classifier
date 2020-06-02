import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import torch


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1,2,0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    
    return ax


def process_image(image):
    
    resize_size = 256, 256
    crop_width, crop_height = 224, 224
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    im = Image.open(image)

    width, height = im.size
    
    n_img = im.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = n_img.size
    
    l = (width-crop_width)/2
    t = (height-crop_height)/2
    r = (width+crop_width)/2
    b = (height+crop_height)/2
    
    c_crop = n_img.crop((l,t,r,b)) 
    
    np_image = np.array(c_crop)
    
    np_image = np_image.transpose((2,0,1))
    
    np_image = np_image/255
    
    
    np_image[0] = ((np_image[0]-means[0])/std[0])
    np_image[1] = ((np_image[1]-means[1])/std[1])
    np_image[2] = ((np_image[2]-means[2])/std[2])
    
    np_image = np_image[np.newaxis,:]
    
    
    tensor_image = torch.from_numpy(np_image)
    tensor_image = tensor_image.float()
    return tensor_image
    
