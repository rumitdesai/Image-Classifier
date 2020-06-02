import argparse

def get_train_input_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory',type = str)
    
    parser.add_argument('--save_dir', type = str, default = './', 
                    help = 'path to save network checkpoints')
    
    parser.add_argument('--arch', type = str, default = 'densenet121', 
                    help = 'Network architecture for classifying images')
    
    parser.add_argument('--learning_rate', type = float, default = 0.003, 
                    help = 'Network architecture for classifying images')
    parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'Specify hidden units to use in intermediate layers')
    parser.add_argument('--epochs', type = int, default = '5', 
                    help = 'Number of iterations to train the network')
    
    parser.add_argument('--gpu', action='store_true')
    
    return parser.parse_args()



def get_predict_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('Path_to_image',type = str)
    
    parser.add_argument('Path_to_checkpoint', type= str)
    
    parser.add_argument('--top_k', type=int, default=5, 
                        help = 'Top K highest probability classes')
    
    parser.add_argument('--category_names',type=str, default='./ImageClassifier/cat_to_name.json', 
                        help='JSON file specifying the mapping from number to category name')
    
    parser.add_argument('--gpu', action='store_true')
    
    return parser.parse_args()