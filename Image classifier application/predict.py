
import os
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np

import command_line
import helper

select_network = {'densenet121':models.densenet121,
                  'densenet169':models.densenet169,
                  'densenet161':models.densenet161,
                  'densenet201':models.densenet201,
                  'vgg11':models.vgg11,
                  'vgg13':models.vgg13,
                  'vgg16':models.vgg16,
                  'vgg19':models.vgg19}

#####################################################################################

def BuildVGG(model_name, num_classes, learning_rate):
    
    
    model = select_network[model_name](pretrained=True)
    
    # Keeping model features frozen
    for parameters in model.parameters():
        parameters.requires_grad=False
        
    model.classifier[6] = nn.Sequential(nn.Linear(model.classifier[6].in_features, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(256, num_classes),
                                        nn.LogSoftmax(dim=1))

    return model

#####################################################################################

def BuildDenseNet(model_name, num_classes, learning_rate):
    
    model = select_network[model_name](pretrained=True)
    
    # Keeping model features frozen
    for parameters in model.parameters():
        parameters.requires_grad=False
        
    
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 512),
                                     nn.ReLU(),
                                     nn.Linear(512,256),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(256,num_classes),
                                     nn.LogSoftmax(dim=1))
    
    return model


build_functions = {'densenet121' : BuildDenseNet,
                   'densenet169': BuildDenseNet,
                   'densenet161': BuildDenseNet,
                   'densenet201': BuildDenseNet,
                   'vgg11': BuildVGG,
                   'vgg13': BuildVGG,
                   'vgg16': BuildVGG,
                   'vgg19': BuildVGG}

#####################################################################################

def Predict(model, image_path, topk, device):
    
    image = helper.process_image(image_path)
    
    image = image.to(device)
    
    probabilities = torch.exp(model(image))
    
    probability, classes = probabilities.topk(topk, dim=1)
    
    return probability, classes

#####################################################################################

def Verdict(prob, classes, category_names):
    
    prob = prob.cpu().data.numpy().squeeze()
    classes = classes.cpu().data.numpy()[0,:]
    
    if os.path.isfile(category_names):
    
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    
        name_list=[]
        
        for it in classes:
            name_list.append(cat_to_name[str(it)])
        
        print(f"Following are the Top {len(name_list)} flower predictions and their respective probabilities")
        for i,j in zip(prob, name_list):
            print(f"There is {i*100} % chance the predicted flower is {j}")
    else:
        print(f"Following are the Top {len(prob)} numerical category class predictions and their respective probabilities")
        for i,j in zip(prob, classes):
            print(f"There is {i*100} % chance the predicted category is {j}")
        
#####################################################################################
def main():
    
    in_args=command_line.get_predict_input_args()
    
    device = 'cuda' if in_args.gpu == True else 'cpu'
    
    #print(in_args)
    
    loaded_state_dict = torch.load(in_args.Path_to_checkpoint)
    
    if 'model_name' not in loaded_state_dict or 'num_classes' not in loaded_state_dict or 'learning_rate' not in loaded_state_dict:
        print('Loaded checkpoint does not capture information to rebuild the model')
        return 1
    
    model_name = loaded_state_dict['model_name'] 
    model = None
    criterion = None
    optimizer = None
    
    model = build_functions[model_name](model_name, loaded_state_dict['num_classes'], loaded_state_dict['learning_rate'])
    
    if model == None:
        print('Failed to rebuild network')
        return 1
    
    
    model.load_state_dict(loaded_state_dict['state_dict'])
    model.class_to_idx = loaded_state_dict['class_to_idx']
    model = model.to(device)
    model.eval()
    
    # Prediction
    #print(device)
    probability, classes = Predict(model, in_args.Path_to_image, in_args.top_k, device)
    
    Verdict(probability, classes, in_args.category_names)
    
        
if __name__ == '__main__':
    main()