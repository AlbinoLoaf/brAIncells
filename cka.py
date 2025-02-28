"""
Author Marius Thomsen inspired by:Repo: https://github.com/numpee/CKA.pytorch

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

        # Register hooks for layers that exist in the model
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

class CKACalculator:
    def __init__(self, model1: nn.Module, model2: nn.Module, dataloader, layers_to_hook=(nn.Conv2d, nn.Linear)):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1, self.model2 = model1.to(self.device).eval(), model2.to(self.device).eval()
        self.dataloader = dataloader
        self.hook_manager1 = HookManager(self.model1, layers_to_hook)
        self.hook_manager2 = HookManager(self.model2, layers_to_hook)

    @torch.no_grad()
    def calculate_cka_matrix(self):
        for images, *_ in tqdm(self.dataloader, desc="Processing CKA"):
            images = images.to(self.device)
            self.model1(images)
            self.model2(images)

        activations1 = self.hook_manager1.get_activations()
        activations2 = self.hook_manager2.get_activations()

        # Store module names for reference
        self.module_names_X = list(activations1.keys())
        self.module_names_Y = list(activations2.keys())

        def gram_matrix(X): return X.matmul(X.t()) # Compute Gram matrix

        def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
            """ Batch-wise computation of HSIC. """
            assert K.size() == L.size() and K.dim() == 3  # (B, N, N)
            B, N, _ = K.shape

            K, L = K.clone(), L.clone()
            # K, L --> K~, L~ by setting diagonals to zero
            K.diagonal(dim1=-1, dim2=-2).fill_(0)
            L.diagonal(dim1=-1, dim2=-2).fill_(0)

            KL = torch.bmm(K, L)
            trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

            middle_term = K.sum(dim=(-1, -2), keepdim=True) * L.sum(dim=(-1, -2), keepdim=True)
            middle_term /= (N - 1) * (N - 2)

            right_term = KL.sum(dim=(-1, -2), keepdim=True) * 2 / (N - 2)

            hsic_value = (trace_KL + middle_term - right_term) / (N**2 - 3*N)
            return hsic_value.squeeze(-1).squeeze(-1)

        cka_matrix = torch.zeros(len(activations1), len(activations2))

        for i, (name1, X) in enumerate(activations1.items()):
            for j, (name2, Y) in enumerate(activations2.items()):
                K = gram_matrix(X.flatten(start_dim=1)).unsqueeze(0) 
                L = gram_matrix(Y.flatten(start_dim=1)).unsqueeze(0)

                hsic_XY = hsic(K, L)
                hsic_XX = hsic(K, K)
                hsic_YY = hsic(L, L)

                cka_matrix[i, j] = hsic_XY / (hsic_XX.sqrt() * hsic_YY.sqrt())

        return cka_matrix