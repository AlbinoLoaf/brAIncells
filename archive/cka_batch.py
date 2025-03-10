"""
inspired by:Repo: https://github.com/numpee/CKA.pytorch

"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torcheeg.models.gnn.dgcnn import Chebynet
from torcheeg.models.gnn.dgcnn import GraphConvolution

class HookManager:
    def __init__(self, model, layers_to_hook=(nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d, GraphConvolution, nn.BatchNorm1d)):
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
        self.hooks = []



def centering(K):
        """Centers the kernel matrix K."""
        n = K.size(0)
        H = torch.eye(n, device=K.device,dtype=torch.double) - 1.0 / n * torch.ones((n, n), device=K.device,dtype=torch.double)
        H = H.double()
        return H @ K @ H


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Computes the HSIC measure."""
    K, L = centering(K), centering(L)
    return torch.trace(K @ L) / ((K.size(0) - 1) ** 2 +1e-8)

def gram_matrix(X):
    return X @ X.t()


class BatchCKACalculator:
    def __init__(self, model1: nn.Module, model2: nn.Module, dataloader, batch_size=64, layers_to_hook=(nn.Conv2d, nn.Linear)):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1, self.model2 = model1.to(self.device).eval(), model2.to(self.device).eval()
        self.dataloader = dataloader
        self.hook_manager1 = HookManager(self.model1, layers_to_hook)
        self.hook_manager2 = HookManager(self.model2, layers_to_hook)
        self.batch_size = batch_size
        

    @torch.no_grad()
    def calculate_cka_matrix(self):
        
        batch_size = self.batch_size
        
        for images, *_ in tqdm(self.dataloader, desc="Processing CKA", disable=True):
            images = images.to(self.device)
            self.model1(images)
            self.model2(images)

        activations1 = self.hook_manager1.get_activations()
        activations2 = self.hook_manager2.get_activations()
        
        self.module_names_X = list(activations1.keys())
        self.module_names_Y = list(activations2.keys())
        
        num_layers = len(self.module_names_X)
        
        cka_matrix = torch.zeros(len(activations1), len(activations2))
        
        for layer_idx_1, layer_idx_2 in [(x,y) for x in range(num_layers) for y in range(num_layers)]:
            
            act1 = list(activations1.values())[layer_idx_1]
            act2 = list(activations2.values())[layer_idx_2]
            
            #print(type(activations1))

            activations_dataset = TensorDataset(act1, act2)
            activations_dl = DataLoader(activations_dataset, batch_size=batch_size, shuffle=False)
            
            num_activations = act1.shape[0]
            
            hsic_XY_agg = 0; hsic_XX_agg = 0; hsic_YY_agg = 0 

            for idx, (X, Y) in enumerate(activations_dl):

                start_idx = (idx * batch_size)

                if start_idx + batch_size <= num_activations:
                    end_idx = start_idx + (batch_size)
                else:
                    end_idx = num_activations

                K = gram_matrix(X.flatten(start_dim=1).double())
                L = gram_matrix(Y.flatten(start_dim=1).double())

                hsic_XY_agg += hsic(K, L)
                hsic_XX_agg += hsic(K, K)
                hsic_YY_agg += hsic(L, L)

            cka_matrix[layer_idx_1, layer_idx_2] = hsic_XY_agg / (hsic_XX_agg.sqrt() * hsic_YY_agg.sqrt())

        # Clear activations to free memory
        self.hook_manager1.clear()
        self.hook_manager2.clear()


        return cka_matrix
    
