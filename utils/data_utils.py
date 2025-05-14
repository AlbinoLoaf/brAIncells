import os
import torch
import pickle
import numpy as np

from torcheeg import transforms
from torcheeg.models import DGCNN
from sklearn.model_selection import train_test_split
from scipy.stats import chisquare,poisson
import utils.graph_utils as gu

node_labels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

#Preprocessing
def band_preprocess(X, preprocessed_data_path):
    """
    Apply band differential entropy preprocessing
    
    ...
    
    Parameters
    -----
    X : torch.Tensor
        Features
    preprocessed_data_path : string
        Path to load preprocessed data from / to save preprocessed data to

    Returns
    -----
    X_bde : torch.FloatTensor
        Data preprocessed using band differential entropy
    
    """
    
    bands = {"delta": [1, 4],"theta": [4, 8],"alpha": [8, 14],"beta": [14, 31],"gamma": [31, 49]}
    if os.path.exists(preprocessed_data_path):

        with open(preprocessed_data_path, "rb") as f:
            X_bde = np.load(f)

    else:
        t = transforms.BandDifferentialEntropy(band_dict=bands)

        X_bde = []
        for i in range(X.shape[0]):

            bde_tmp = t(eeg=X[i])
            X_bde.append(bde_tmp)

        X_bde = [x["eeg"] for x in X_bde]

        with open(preprocessed_data_path, "wb") as f:
            np.save(f, X_bde)

    X_bde = torch.FloatTensor(X_bde)     
    return X_bde


def split_data(X, y, has_val_set=False, seed=42):
    
    if has_val_set:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
        
        return (X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

        return (X_train, y_train, X_test, y_test)
    
    
def chi2(node_counts_all,comp_set,ddof=0):
    observed_counts = np.array(node_counts_all)
    if abs(sum(observed_counts) - sum(comp_set)) < 0.0001:
            expected_counts = np.array(comp_set) #synthetic data 

    elif sum(observed_counts)<sum(comp_set):
        expected_counts=np.array(gu.shrink(comp_set,sum(observed_counts)))
    else:
        expected_counts=np.array(gu.grow(comp_set,sum(observed_counts)))
    
    chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts,ddof=ddof)
    print(f"Chi-squared statistic: {chi2_stat:.3g}")
    print(f"P-value: {p_value:.3g}\n")
    
    
def getlikelihood(lst_count,total):
    return [round((lst_count[i]/total)*100,2) for i in range(len(lst_count))] 


def getuniformcomp(lst_count,total):
    uniform=(1/len(node_labels))
    return [round(((lst_count[i]/total)/uniform)*100-100,2) for i in range(22)]


def statistical_tests(counts, untrained_counts, metric_dict, metric_name, include_untrained=True):
    
    total = sum(counts)
   
    print(f"-----Tests for metric: {metric_name}-----\n")
    print("---Uniform test---")
    chi2(counts,np.full(22,total/22))
    
    #Likelihood
    likelihood=getlikelihood(counts,total)
    print("----Likelyhood-----")
    print(likelihood)
    #uniform comparison
    print("----Uniform comparison-----")
    uniform_comp=getuniformcomp(counts,total)
    print(uniform_comp)
    print(f"Total/Differential")
    for i in range(len(node_labels)):
        print(f" {node_labels[i]}: {likelihood[i]}% : {uniform_comp[i]}% ")
        
    print("---Poisson test---")
    metric_sorted, _ = gu.get_sorted_metrics(metric_dict, node_labels)
    lambda_est = np.mean(metric_sorted)
    expected_poisson = [max(int(poisson.pmf(k, lambda_est) * sum(metric_sorted)),1) for k in range(22)]
    chi2(metric_sorted, expected_poisson,ddof=1)
    
    if include_untrained:
        print("---Is altered from untrained?---")
        chi2(counts,untrained_counts)
        unt_likelihood=getlikelihood(untrained_counts,total)
        unt_uniform_comp=getuniformcomp(untrained_counts,total)
        print("---Untrained likelyhood---")
        for i in range(22):
            print(f" {node_labels[i]}: {unt_likelihood[i]}% : {unt_uniform_comp[i]}% ")
        


