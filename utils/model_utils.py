#Matrix 
import numpy as np
import random
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheeg.models import DGCNN
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import utils.graph_utils as gu

def threshold(mat, thresh=0.2):
    """ 
    Helper function for get_adj_mat, Sets all entries in a matrix below the threshold to 0

    ...

    Parameters
    -----
    mat : two numpy arrays as a matrix
        Torch.tensor as an matrix for numpy to cut off all low values
    thresh : float
        Threshold for the cut off

    Returns
    -----
    mat : two numpy arrays as a matrix
        returns the matrix it recieved but with all  values below the thresh set to zero
    """
    mat[mat < thresh] = 0.0
    return mat

def get_adj_mat(model, thresh=0.2):
    """ 
    Extracts the adjacency matrix from a DGCNN model and normalise it, cutting noise

    ...

    Parameters
    -----
    model : torcheeg.models DGCNN 
        neural network model with a learned adjecency matrix
    thresh : float
        Threshold for the cut off

    Returns
    ----- 
     A : two numpy arrays as a matrix
        The adjacency matrix from the network
    """
    A=F.relu(model.A)
    N=A.shape[0]
    A=A*(torch.ones(N,N)-torch.eye(N,N))
    A=A+A.T
    A=threshold(A.detach(),thresh)
    return A

def get_preds(model, X):
    """
    Get predicted class from predicted class probabilities
    
    ...
    
    Parameters
    -----
    model: torcheeg.models DGCNN
    X: torch.FloatTensor
        Data to predict classes for
    
    Returns
    -----
    preds: np.ndarray
        Predicted class int labels
    """
    _, preds = torch.max(model(X), 1)
    return preds.detach().numpy()


def eval_model(model, X_train, y_train, X_test, y_test):

    preds_train = get_preds(model, X_train)
    preds_test = get_preds(model, X_test)
        
    y_train_npy = y_train.numpy() ; y_test_npy = y_test.numpy()
    
    acc_train = accuracy_score(y_train_npy, preds_train)
    acc_test = accuracy_score(y_test_npy, preds_test)
    
    return acc_train, acc_test, acc_train-acc_test


def get_all_models(model_dict):
    
    models = []
    
    for seed_val in model_dict.keys():
        
        curr_models_dict = model_dict[seed_val]
        
        for chan_val in curr_models_dict.keys():
            
            curr_models = curr_models_dict[chan_val]
            models.append(curr_models)
        
    return models

def avg_performance(models, X_train, y_train, X_test, y_test, prints=True):
    
    N = len(models)
    train_accs = np.zeros(N); test_accs = np.zeros(N); difs = np.zeros(N)
    
    for i in range(N):
        curr_train, curr_test, curr_dif = eval_model(models[i], X_train, y_train, X_test, y_test)
        train_accs[i] = curr_train; test_accs[i] = curr_test; difs[i] = curr_dif
        
    train_avg = np.mean(train_accs); test_avg = np.mean(test_accs); difs_avg = np.mean(difs)
    train_std = np.std(train_accs); test_std = np.std(test_accs); difs_std = np.std(difs)
    
    if prints:
        print(f"Average models training accuracy: {train_avg:.3f}")
        print(f"Average models test accuracy: {test_avg:.3f}")
        print(f"Average models training-test accuracy difference: {difs_avg:.3f}")

        print(f"Models training accuracy SD: {train_std:.3f}")
        print(f"Models test accuracy SD: {test_std:.3f}")
        print(f"Models training-test accuracy difference SD: {difs_std:.3f}")
    
    return train_avg, test_avg, difs_avg, train_std, test_std, difs_std 


def read_saved_models(saved_path):    
    main_folder_files = []
    for folder in os.listdir(saved_path):
        if "gpuerror" not in folder:
            file_names = os.listdir(saved_path + folder)
            folder_seed = int(folder.split("_")[-1])
            main_folder_files.append([folder, file_names, folder_seed])

    model_objects = dict()
    for folder_name, folder_files, seed in main_folder_files:
        #print(f"for seed = {seed}")
        model_objects[seed] = dict()
        for file_name in folder_files:

            if file_name[:5] == "model":
                #print(f"file_name: {file_name} - MODEL file")
                chans = int(file_name.split("_")[2])
                full_model_path = saved_path + "/" + folder_name + "/" + file_name
                
                curr_mod = DGCNN(in_channels=5, num_electrodes=22, 
                              hid_channels=chans, num_layers=2, num_classes=5)
                
                model_weights = torch.load(full_model_path, map_location=torch.device('cpu'))
                curr_mod.load_state_dict(model_weights)
                model_objects[seed][chans] = curr_mod
                
    return model_objects


def metrics_by_chans(model_dict, get_external=False):
    
    model_by_chans = dict()
    for curr_seed in model_dict.keys():
        for n_chans in model_dict[curr_seed].keys():
            if n_chans not in model_by_chans.keys():
                model_by_chans[n_chans] = []
            model_by_chans[n_chans].append(model_dict[curr_seed][n_chans])
    
    barycenters_by_chans = dict()
    sims_by_chans = dict()
    isomorphism_by_chans = dict()
    geds_by_chans = dict()
    for n_chans in model_by_chans.keys():
        curr_models = model_by_chans[n_chans]
        
        barycenters, sims = gu.get_graph_metrics_internal(curr_models, prints=False)
        barycenters_by_chans[n_chans] = barycenters
        sims_by_chans[n_chans] = sims
        
        if get_external:
            iso, geds = gu.get_graph_metrics_external(curr_models, prints=False)
            isomorphism_by_chans[n_chans] = iso
            geds_by_chans[n_chans] = geds

    if get_external:
        return model_by_chans, barycenters_by_chans, sims_by_chans, isomorphism_by_chans, geds_by_chans
    else:
        return model_by_chans, barycenters_by_chans, sims_by_chans

def model_metrics(model, X_train, y_train, X_test, y_test, X_val=None, y_val=None,plots=True):
    """
    Display model metrics accuracy, f1 score and confusion matrix for the training, validation
    (if applicable) and test sets
    
    ...
    
    Parameters
    -----
    model: torcheeg.models DGCNN
    X_train, X_val, X_test: torch.FloatTensor
        Input features for the train, validation (if applicable) and test sets
    y_train, y_val, y_test: torch.FloatTensor
        Output labels for the train, validation (if applicable) and test sets
    """
    preds_train = get_preds(model, X_train)
    preds_test = get_preds(model, X_test)
    
    labels = ["feet","left_hand","right_hand","tongue"]
    
    y_train_npy = y_train.numpy()
    y_test_npy = y_test.numpy()
    
    acc_train = accuracy_score(y_train_npy, preds_train)
    acc_test = accuracy_score(y_test_npy, preds_test)
    
    f1_train = f1_score(y_train_npy, preds_train, average="macro")
    f1_test = f1_score(y_test_npy, preds_test, average="macro")
    
    if plots:
        print(f"Acc train: {acc_train}")
        print(f"Acc test: {acc_test}")
        
        print(f"F1 train: {f1_train}")
        print(f"F1 test: {f1_test}")
        conf_mat_train = confusion_matrix(y_train, preds_train)
        disp_train = ConfusionMatrixDisplay(confusion_matrix=conf_mat_train, display_labels=labels)

        ax = disp_train.plot()
        ax.ax_.set_title("Confusion matrix - Training set")

        plt.show()
        
        conf_mat_test = confusion_matrix(y_test, preds_test)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=conf_mat_test, display_labels=labels)

        # Plot and set title
        ax = disp_test.plot()  # returns (figure, axes)
        ax.ax_.set_title("Confusion matrix - Test set")

        plt.show()
    return acc_train,f1_train,acc_test,f1_test

def confusiong_avg(modlist, X_train, y_train, X_test, y_test, plots=True):
    conf_mats_train = np.zeros((4, 4))
    conf_mats_test = np.zeros((4, 4))

    for i in range(len(modlist)):
        preds_train = get_preds(modlist[i][0], X_train)
        preds_test = get_preds(modlist[i][0], X_test)
        conf_mats_train += confusion_matrix(y_train, preds_train)
        conf_mats_test += confusion_matrix(y_test, preds_test)

    conf_mats_train /= len(modlist)
    conf_mats_test /= len(modlist)

    if plots:
        labels = ["feet", "left_hand", "right_hand", "tongue"]

        disp_train = ConfusionMatrixDisplay(confusion_matrix=conf_mats_train, display_labels=labels)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=conf_mats_test, display_labels=labels)

        fig, ax = plt.subplots()
        disp_train.plot(ax=ax, values_format=".1f")
        ax.set_title("Confusion matrix - Train set")
        plt.show()

        fig, ax = plt.subplots()
        disp_test.plot(ax=ax, values_format=".1f")
        ax.set_title("Confusion matrix - Test set")
        plt.show()

    return conf_mats_train, conf_mats_test


def multi_parameter_mod(param_list, n_models):
    # TO DO: take into account seeds. Right now the seed_list doesn't do anything
    # all combinations of model indexes between two parameter sets
    # ex all model combinations between models with 8 hidden neurons and models with 16 hidden neurons
    
    
    combs_external =[
        (i, j)
        for i in range(n_models)
        for j in range(n_models, 2 * n_models)
        ]
    #combs_external = list(itertools.product([x for x in range(n_models)], [x+n_models for x in range(n_models)]))
    # all combinations of parameter values, in this case number of hidden neurons
    param_combs = [
        (param_list[i], param_list[j]) 
        for i in range(0,   len(param_list)) 
        for j in range(i+1, len(param_list))]
    
    models_dict, bary_dict, sim_dict, edit_dists_internal = internal_dict(param_list)
    edit_dists_external = lst_to_dict(param_combs)   #Graph metric dict n for GED between different param models
    
    # train n_models models with each number of hidden neurons specified in the param_list
    for n_chans in param_list:
        a=0
        #curr_model = [x[0] for x in train_models(DGCNN, TrainNN, n_chans, seed_list, num_models = n_models,prints=plot, new=False)]
        #models_dict[n_chans].extend(curr_model)
    
    # calculate all metrics for models with the same number of hidden neurons
    for n_chans in param_list:
        models = models_dict[n_chans]
        bary, sim, _, ed = gu.get_graph_metrics(models, prints=False)
        bary_dict[n_chans].extend(bary)
        sim_dict[n_chans].extend(sim)
        edit_dists_internal[n_chans].extend(ed)
    
    # calculate edit distance between models with different number of hidden neurons
    for param_comb in param_combs:
        for ext_comb in combs_external:
            model1_idx = ext_comb[0]; model2_idx = ext_comb[1]
            model1 = models_dict[param_comb[0]][model1_idx]
            model2 = models_dict[param_comb[1]][model2_idx-n_models]
            G1 = gu.make_graph(mu.get_adj_mat(model1))
            G2 = gu.make_graph(mu.get_adj_mat(model2))
            ed_external = next(nx.optimize_graph_edit_distance(G1, G2))
            edit_dists_external[param_comb].append(ed_external)

    return models_dict, bary_dict, sim_dict, edit_dists_internal, edit_dists_external




class TrainNN():
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, model, train_loader, learning_rate, path,name,has_val_set=False,
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
        torch.save(model.state_dict(), path)

        if has_val_set:
            losses = np.array([losses_train, losses_val])
        else:
            losses = np.array(losses_train)

        run.finish()

        return model, losses