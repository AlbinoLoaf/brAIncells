import numpy as np
import torch
import torch.nn as nn
from torcheeg.models.gnn.dgcnn import GraphConvolution

class HookManager:
    def __init__(self, model, 
                 layers_to_hook=(nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d, GraphConvolution, nn.BatchNorm1d)):
        self.activations = {}
        self.hooks = []

        existing_layers = {type(layer) for layer in model.modules()}
        valid_layers = tuple(layer for layer in layers_to_hook if layer in existing_layers)

        if not valid_layers:
            print("[WARNING] No matching layers found in the model.")

        for name, layer in model.named_modules():
            if isinstance(layer, valid_layers):
                self.hooks.append(layer.register_forward_hook(self._store_activation(name)))

    def _store_activation(self, layer_name):
        def hook(module, _, output):
            self.activations[layer_name] = output.detach()
        return hook

    def get_activations(self):
        return self.activations

    def clear(self):
        self.activations.clear()
        for hook in self.hooks:
            hook.remove()
        #self.hooks = []



def centering(K):
        """Centers the kernel matrix K."""
        n = K.size(0)
        H = torch.eye(n, device=K.device) - 1.0 / n * torch.ones((n, n), device=K.device,)
        H = H
        return H @ K @ H


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Computes the HSIC measure."""
    K, L = centering(K), centering(L)
    return torch.trace(K @ L) / ((K.size(0) - 1) ** 2 +1e-8)

def gram_matrix(X):
    return X @ X.t()


class CKACalculator:
    def __init__(self, model1: nn.Module, model2: nn.Module, dataloader, 
                 layers_to_hook=(nn.Conv2d, nn.Linear,nn.Conv1d)):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1, self.model2 = model1.to(self.device).eval(), model2.to(self.device).eval()
        self.dataloader = dataloader
        self.hook_manager1 = HookManager(self.model1, layers_to_hook)
        self.hook_manager2 = HookManager(self.model2, layers_to_hook)

    @torch.no_grad()
    def calculate_cka_matrix(self,data):     
        """
        Calculates the cka for the model attatched, returns a cka matrix (tensor)

        ....

        Parameters 
        -----
        self : class
            Passing the class to the function
        data : tensor matrix floats
            a minimum 500 datapoint longs dataset do not cut off. this function will handle all manipulations it needs

        Returns
        ----
        cka_matrix : Tensor 
            Matrix of floats for the CKA matrix 
        """   
        x_data = torch.stack([data[i][0] for i in range(len(data))])
        x_data = x_data[:500]
        self.model1(x_data)
        self.model2(x_data)

        activations1 = self.hook_manager1.get_activations()
        activations2 = self.hook_manager2.get_activations()


        if not activations1 or not activations2:
            print("[ERROR] No activations were recorded. Check if the layers are hooked properly.")
            return torch.zeros(0, 0) 
        
        self.module_names_X = list(activations1.keys())
        self.module_names_Y = list(activations2.keys())

        cka_matrix = torch.zeros(len(activations1), len(activations2))

        for i, (name1, X) in enumerate(activations1.items()):
            for j, (name2, Y) in enumerate(activations2.items()):
                K = gram_matrix(X.flatten(start_dim=1))
                L = gram_matrix(Y.flatten(start_dim=1))

                hsic_XY = hsic(K, L)
                hsic_XX = hsic(K, K)
                hsic_YY = hsic(L, L)

                cka_matrix[i, j] = hsic_XY / (hsic_XX.sqrt() * hsic_YY.sqrt())
        # Clear activations to free memory
        self.hook_manager1.clear()
        self.hook_manager2.clear()


        return cka_matrix
    
    def test_cka(self, num_test,data):
        Matrix_field_val = np.zeros(num_test)
        for i in range(num_test):
            cka_output = self.calculate_cka_matrix(self,data)
            Matrix_field_val[i]=cka_output[0][4]
        print(f"Standard deviation: {np.std(Matrix_field_val)}")
        print(f"Array max: {max(Matrix_field_val)}\nArray min: {min(Matrix_field_val)}\nArray mean: {np.mean(Matrix_field_val)}")