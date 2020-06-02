
import os

import torch 
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
#from PIL import Image
import numpy as np

# Local imports
from workspace_utils import active_session
import command_line
import helper


networks= {'densenet121':models.densenet121,
           'densenet169':models.densenet169,
           'densenet161':models.densenet161,
           'densenet201':models.densenet201,
           'vgg11':models.vgg11,
           'vgg13':models.vgg13,
           'vgg16':models.vgg16,
           'vgg19':models.vgg19}


#####################################################################################

def BuildVGG(in_args, num_classes):
    
    model = networks[in_args.arch](pretrained = True)
    
    # Keeping model features frozen
    for parameters in model.parameters():
        parameters.requires_grad=False
        
    model.classifier[6] = nn.Sequential(nn.Linear(model.classifier[6].in_features, in_args.hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(in_args.hidden_units, num_classes),
                                        nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=in_args.learning_rate)
    
    return model, criterion, optimizer


#####################################################################################

def BuildDenseNet(in_args, num_classes):
    
    model = networks[in_args.arch](pretrained = True)
    
    # Keeping model features frozen
    for parameters in model.parameters():
        parameters.requires_grad=False
        
    
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, in_args.hidden_units),
                                     nn.ReLU(),
                                     nn.Linear(in_args.hidden_units, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(256,num_classes),
                                     nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    
    return model, criterion, optimizer


##################################################################################################

# train the network
def TrainNetwork(model, optimizer, criterion, traindataloader, validationdataloader, device, epochs):
    running_loss=0
    steps =0

    train_losses, validation_losses=[],[]
    with active_session():
        for e in range(epochs):
            running_loss=0
            for images, labels in traindataloader:
                steps+=1
                images, labels = images.to(device), labels.to(device)
        
                # Reset optimizer which might have accumulated gradients
                optimizer.zero_grad()
        
                # feed forward with training input images
                logps = model.forward(images)
        
                # calculate the loss
                loss = criterion(logps, labels)
        
                # backpropgation to calculate gradients pertaining to classifier
                loss.backward()
        
                # Update the classiier weights
                optimizer.step()
        
                running_loss += (loss.item()*images.size(0))
            else:
                validation_loss=0
                accuracy=0
            
                with torch.no_grad():
                    model.eval()
                    for v_images, v_labels in validationdataloader:
                        v_images, v_labels = v_images.to(device), v_labels.to(device)
                    
                        v_logps = model.forward(v_images)
                        validation_loss += (criterion(v_logps, v_labels)*v_images.size(0))
                    
                        # Validation Accuracy
                        ps = torch.exp(v_logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                model.train()
                train_losses.append(running_loss/len(traindataloader))
                validation_losses.append(validation_loss/len(validationdataloader))
            
                print(f"Epoch {e+1}/{epochs}.."
                      f"Train loss: {train_losses[-1]:.5f}.."
                      f"Validation loss: {validation_losses[-1]: .5f}.."
                      f"Validation accuracy: {accuracy/ len(validationdataloader):.5f}")
    
    model.eval()
    return model

##################################################################################################

build_functions = {'densenet121' : BuildDenseNet,
                   'densenet169': BuildDenseNet,
                   'densenet161': BuildDenseNet,
                   'densenet201': BuildDenseNet,
                   'vgg11': BuildVGG,
                   'vgg13': BuildVGG,
                   'vgg16': BuildVGG,
                   'vgg19': BuildVGG}


def main():
    
    in_args = command_line.get_train_input_args()
    #print(in_args)
    
    # Constant parameters
    device = 'cuda' if in_args.gpu == True else 'cpu'
    epochs = in_args.epochs
    print(device)
    
    # Specify training directory, testing and validation directory
    train_dir = in_args.data_directory+'/train' if os.path.isdir(in_args.data_directory+'/train') == True else None 
    if train_dir == None:
        print(f"Training data not available at: {in_args.data_directory}")
        return 1
    
    test_dir = in_args.data_directory+'/test' if os.path.isdir(in_args.data_directory+'/test') == True else None 
    valid_dir = in_args.data_directory+'/valid' if os.path.isdir(in_args.data_directory+'/valid') == True else None 
    
    if train_dir == None and valid_dir == None:
        print(f"No testing or validation data available at: {in_args.data_directory}")
        return 1
    
    # Defining transforms for training, testing and validation
    train_data_transform = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],
                                                                                          [0.229, 0.224, 0.225])])
    
    validation_data_transform = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
    
    test_data_transform = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    
    # Loading dataset with ImageFolder
    train_image_data = datasets.ImageFolder(train_dir, transform = train_data_transform)
    validation_image_data = datasets.ImageFolder(valid_dir, transform = validation_data_transform)
    test_image_data = datasets.ImageFolder(test_dir, transform = test_data_transform)
    
    # Defining dataLoaders
    traindataloader = DataLoader(train_image_data, batch_size=32, shuffle=True)
    validationdataloader = DataLoader(validation_image_data, batch_size=32, shuffle=True)
    testdataloader = DataLoader(test_image_data, batch_size=32, shuffle=True)
    
    ########################################################################################################
    
    if in_args.arch not in networks or in_args.arch not in build_functions:
        print(f"This application does not support {in_args.arch}.\nSupported networks are:\n{[n for n in networks.keys()]}\n"
              f"Please select from above listed networks")
        return 1
    
    model, criterion, optimizer = build_functions[in_args.arch](in_args, 102)
    model = model.to(device)
    
    trained_model = TrainNetwork(model, optimizer, criterion, traindataloader, traindataloader, device, epochs)
    
    
    # Save the model to a checkpoint
    state_dict = {'model_name': in_args.arch,
                  'num_classes': 102,
                  'learning_rate': in_args.learning_rate,
                  'class_to_idx': train_image_data.class_to_idx,
                  'state_dict': trained_model.state_dict()}
    
    torch.save(state_dict, in_args.save_dir+'R'+in_args.arch+'.pth')
    
    
if __name__ == "__main__":
    main()