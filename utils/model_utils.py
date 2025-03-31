#Matrix 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
    
    if X_val != None:
        preds_val = get_preds(model, X_val)
        y_val_npy = y_val.numpy()
        acc_val = accuracy_score(y_val_npy, preds_val)
        f1_val = f1_score(y_val_npy, preds_val, average="macro")
        print(f"Acc val: {acc_val}")
        print(f"F1 val: {f1_val}")
        
        conf_mat_val = confusion_matrix(y_val, preds_val)
        disp_val = ConfusionMatrixDisplay(confusion_matrix=conf_mat_val, display_labels=labels)

        ax = disp_val.plot()
        ax.ax_.set_title("Confusion matrix - Validation set")

        plt.show()
        
    y_train_npy = y_train.numpy()
    y_test_npy = y_test.numpy()
    
    acc_train = accuracy_score(y_train_npy, preds_train)
    acc_test = accuracy_score(y_test_npy, preds_test)
    
    f1_train = f1_score(y_train_npy, preds_train, average="macro")
    f1_test = f1_score(y_test_npy, preds_test, average="macro")
    
    print(f"Acc train: {acc_train}")
    print(f"Acc test: {acc_test}")
    
    print(f"F1 train: {f1_train}")
    print(f"F1 test: {f1_test}")
    
    if plots:
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


def confusiong_avg(modlist,X_train, y_train, X_test, y_test,plots=True):
    conf_mats_train=np.zeros((4,4))
    conf_mats_test=np.zeros((4,4))

    for i in range(len(modlist)):
        preds_train = get_preds(modlist[i][0], X_train)
        preds_test = get_preds(modlist[i][0], X_test)
        conf_mats_train += confusion_matrix(y_train, preds_train)
        conf_mats_test  += confusion_matrix(y_test, preds_test)
    conf_mats_train /= len(modlist)
    conf_mats_test /= len(modlist)
    if plots:
        labels = ["feet","left_hand","right_hand","tongue"]
       
        disp_train = ConfusionMatrixDisplay(confusion_matrix=conf_mats_train, display_labels=labels)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=conf_mats_test, display_labels=labels)

        # Plot and set title
        ax = disp_train.plot()  # returns (figure, axes)
        ax.ax_.set_title("Confusion matrix - Train set")

        plt.show()
        ax = disp_test.plot()  # returns (figure, axes)
        ax.ax_.set_title("Confusion matrix - Test set")

        plt.show()
    return conf_mats_train, conf_mats_test    

class TrainNN():
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, model, train_loader, learning_rate, 
                    path,name,has_val_set=False,val_loader=None,w_decay=1e-4, epochs=100, prints=True, modrun=0):
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
            project = "testing_200",
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

        
        torch.save(model.state_dict(), f'{path}/{name}{modrun}.pth')

        if has_val_set:
            losses = np.array([losses_train, losses_val])
        else:
            losses = np.array(losses_train)

        with open(f"{path}/metrics{modrun}.npy", "wb") as f:
            np.save(f, losses)

        run.finish()

        return model, losses