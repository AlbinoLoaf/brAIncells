import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheeg.models import DGCNN
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_all_models(model_dict):
    """
    Extract model objects from seed channel dict to list
    
    ...
    
    Parameters
    -----
    model_dict: dict
        Dictionary of the form dict[seed][channels] containing 
        the trained model objects (torcheeg.models DGCNN objects)
    
    Returns
    -----
    models: list
        List of all the model objects from the dictionary
    """
    
    models = []
    
    for seed_val in model_dict.keys():
        
        curr_models_dict = model_dict[seed_val]
        
        for chan_val in curr_models_dict.keys():
            
            curr_models = curr_models_dict[chan_val]
            models.append(curr_models)
        
    return models






def read_all_models(saved_path, test_parameter, baseline):
    """
    Load all models organized by tested hyperparameter.
    
    Returns:
        models[param_value] = [model1, model2, ...]
    """
    models = {}

    for folder in os.listdir(saved_path):
        if "gpuerror" in folder:
            continue

        folder_path = os.path.join(saved_path, folder)
        seed_str = folder.split("_")[-1]
        try:
            seed = int(seed_str)
        except ValueError:
            print(f"Invalid folder name: {folder_path}")
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.startswith("model"):
                continue

            full_path = os.path.join(folder_path, file_name)
            saveobj = torch.load(full_path, map_location='cpu')
            params = saveobj['params']

            # Check if baseline matches all except test param
            is_valid = True
            for key in baseline.keys():
                if key == test_parameter:
                    continue
                if baseline[key] != params[key]:
                    is_valid = False
                    break

            if not is_valid:
                continue

            test_value = params[test_parameter]

            # Initialize model with correct parameters
            init_params = baseline.copy()
            init_params[test_parameter] = test_value
            model = DGCNN(**init_params)
            model.load_state_dict(saveobj['model_state'])
            model.params = params
            model.eval()

            # Store in dict
            if test_value not in models:
                models[test_value] = []
            models[test_value].append(model)

    return models


# def read_saved_models(saved_path):
#     """
#     Read saved model objects from specified path 
    
#     ...
    
#     Parameters
#     -----
#     saved_path: str
#         Path where model objects are saved
    
#     Returns
#     -----
#     model_objects : dict
#         Dictionary containing read model objects in the form dict[seed][n_chans]
#         (where each dict[seed] contains a dictionary organised by number of channels)
#     """
#     main_folder_files = []
#     for folder in os.listdir(saved_path):
#         if "gpuerror" not in folder:
#             file_names = os.listdir(saved_path + folder)
#             folder_seed = int(folder.split("_")[-1])
#             main_folder_files.append([folder, file_names, folder_seed])

#     model_objects = dict()
#     for folder_name, folder_files, seed in main_folder_files:
#         model_objects[seed] = dict()
#         for file_name in folder_files:

#             if file_name[:5] == "model":
                
#                 chans = int(file_name.split("_")[2])
#                 full_model_path = saved_path + "/" + folder_name + "/" + file_name
                
#                 def findmod(in_channels=5, num_electrodes=22,hid_channels=8, num_layers=2, num_classes=5):
#                     return DGCNN(in_channels=in_channels, num_electrodes=num_electrodes, 
#                               hid_channels=hid_channels, num_layers=num_layers, num_classes=num_classes)
#                 param_dict = {chans:"hid_channels=8"}
#                 curr_mod = findmod(param_dict[param])
                
#                 model_weights = torch.load(full_model_path, map_location=torch.device('cpu'))
#                 curr_mod.load_state_dict(model_weights)
#                 model_objects[seed][chans] = curr_mod
                
#     return model_objects


class TrainNN():
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, model, param_dict, train_loader, learning_rate, path, name,has_val_set=False,
                    val_loader=None,w_decay=1e-4, epochs=100, prints=True, modrun=0):
        '''
        Trains a given torch.nn model
        
        ...

        Parameters
        -----------
        model : torch.nn
              model object to train
        train_loader : torch.utils.data.DataLoader
              training data of the type torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
              validation data of the type torch.utils.data.DataLoader
        learning_rate : float
              training hyperparameter that decides how big each optimization step will be
        pth : string
              path to save model artifact and metrics at
        name : string
              name of the model artifact
        w_decay : float
              weight decay, regularization training hyperparameter
        epochs : int
              number of epochs to train for
        prints : bool
              whether debug prints should be printed or not
        modrun : int
              model id for further comparison using CKA matrices

        Return
        -----------
        model : torch.nn
              model object that can be used for prediction
        losses : np.ndarray
              numpy array where losses[0] is the training loss history and losses[1] 
              is the validation loss history
        '''

        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)

        highest_train_accuracy = 0.0
        highest_val_accuracy = 0.0
        
        losses_train = []; losses_val = []

        run = wandb.init(
            project = "training_1000",
            name="dgcnn",
            config={
                "learning_rate":learning_rate,
                "w_decay":w_decay,
                "modrun":modrun, 
                "epochs":epochs
            }
        )

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0; running_loss_val = 0.0
            correct = 0; correct_val = 0
            total = 0; total_val = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct/total
            losses_train.append(epoch_loss)
            
            if has_val_set:
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss_val += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                epoch_loss_val = running_loss_val / len(val_loader.dataset)
                epoch_accuracy_val = correct_val/total_val
                losses_val.append(epoch_loss_val)
                if epoch_accuracy_val> highest_val_accuracy:
                      highest_val_accuracy = epoch_accuracy_val
                        
            if epoch_accuracy > highest_train_accuracy:
                highest_train_accuracy = epoch_accuracy

            if prints:
                if has_val_set:
                    print(f"Epoch {epoch+1}/{epochs}, Train loss: {epoch_loss:.4f}, Train acc: {(epoch_accuracy*100):.2f}%" +
                         f"| Val loss: {epoch_loss_val:.4f}, Val acc: {(epoch_accuracy_val*100):.2f}%")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train loss: {epoch_loss:.4f}, Train acc: {(epoch_accuracy*100):.2f}") 
            
            if has_val_set:
                run.log({"train_loss":epoch_loss,
                     "train accuracy":epoch_accuracy*100,
                     "eval loss":epoch_loss_val,
                     "eval acc":epoch_accuracy_val*100
                     }, commit=True)
            else:
                run.log({"train_loss":epoch_loss,
                     "train accuracy":epoch_accuracy*100,
                     }, commit=True)
        
        
        if has_val_set:
            print(f"Highest Train Accuracy {(highest_train_accuracy*100):.2f}\nHighest val Accuracy {(highest_val_accuracy*100):.2f}")
        else:
            print(f"Highest Train Accuracy {(highest_train_accuracy*100):.2f}")

        
        print(f"[TrainNN.train_model] : Saving model at {path}")
        #torch.save(model.state_dict(), path)
        torch.save({
            'model_state': model.state_dict(),
            'params': param_dict
        },path)

        if has_val_set:
            losses = np.array([losses_train, losses_val])
        else:
            losses = np.array(losses_train)

        run.finish()

        return model, losses
    
    
def train_models(modeltrainer, dict_model_arch, dict_training, dict_model_meta, path="../artifacts"):
    """
    Training a model with random initialisation but consitent parameters. 
    
    Hyper parameters should be set outside of this function 
    
    path and model_names are both needed parameters that need to be defined outside of this function. 
    
    Path is to where your artifacts are located and model_name is what the model is called 
    
    ...
    
    Parameters
    -----------
    model : nn.modules
        The model being trained
    modeltrainer : training class
        a class for training the model provided should return a trained model
    num_models : int 
        Default 1, how many models it trains
    new : bool
        Weather it should attempt to use saved models
    """
    mods = []
    for i in range(dict_model_meta["amount"]):   
            tmp_mod = dict_model_arch["model"](in_channels=dict_model_arch["in_channels"], 
                                            num_electrodes=dict_model_arch["num_electrodes"], 
                                            hid_channels=dict_model_arch["hid_channels"], 
                                            num_layers=dict_model_arch["num_layers"],
                                            num_classes=dict_model_arch["num_classes"])
            
            mod_name=dict_model_meta["name"]
            
            model_path=f"{path}/{mod_name}_{i}.pth"

            if dict_model_meta["amount"]>1:
                print(f"Model {i+1}")
                
            if not os.path.exists(model_path):
                    print(f"[train_models]: Could not resolve path: {model_path}")
                    dict_model_meta["new_models"]=True
                
            if dict_model_meta["new_models"] or not os.path.exists(model_path):
                
                print(f"[train_models]: Training new models")
                trainer = modeltrainer()
                
                mods.append(trainer.train_model(tmp_mod, dict_model_arch, train_loader, 
                                                    path=model_path,
                                                    name=mod_name,
                                                    has_val_set=False,
                                                    val_loader=None,
                                                    learning_rate=dict_training["lr"],
                                                    w_decay=dict_training["w_decay"],
                                                    epochs=dict_training["epochs"], 
                                                    prints=dict_model_meta["plot"],
                                                    modrun=i))   
            else:
                print(f"[train_models]: Loading models from {model_path}")
                tmp_mod.load_state_dict(torch.load(model_path))
                tmp_mod.eval()
                mods.append([tmp_mod,[]])
    return mods



def model_by_param(model_dict, param):
    """
    Destributes each seed of a given model into channels
    ...
    
    Parameters
    -----
    model_dict: dict
        Dictionary of the form dict[seed][channels] containing 
        the trained model objects (torcheeg.models DGCNN objects)
    param: string
        A string that describes the metric that are in file name
    
    Returns
    -----
    model_by_chans, barycenters_by_chans, sims_by_chans: dict
        Dictionary containing graph metrics of the form dict[n_chans]
    isomorphism_by_chans, geds_by_chans: dict
        Dictionary containing GED and isomorphism check results of the form dict[n_chans]
        Only returned when get_external is set to True
    """
    
    model_by_param = dict()
    for curr_seed in model_dict.keys():
        for curr_param in model_dict[curr_seed].keys():
            if curr_param not in model_by_param.keys():
                model_by_param[curr_param] = []
            model_by_param[curr_param].append(model_dict[curr_seed][curr_param])
    return model_by_param