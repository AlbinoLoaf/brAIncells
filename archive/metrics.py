import torch
from torchmetrics import Metric


class AccumTensor(Metric):
    def __init__(self, default_value: torch.Tensor):
        super().__init__()

        self.add_state("val", default=default_value, dist_reduce_fx="sum")

    def update(self, input_tensor: torch.Tensor):
        self.val += input_tensor

    def compute(self):
        return self.val
    
    
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


def get_graph_metrics_internal(mod_list, prints=False):

    barycenters = []; simrank_similarities = []
    
    # Barycenters and simrank similarity
    for i in range(len(mod_list)):
        
        curr_adj = mu.get_adj_mat(mod_list[i])
        if nx.is_connected(make_graph(curr_adj)):
            curr_barycenter = get_barycenter(curr_adj)
            barycenters.append(curr_barycenter)
        else:
            print("graph not connected")
        G = make_graph(curr_adj)
        sim = get_simrank_similarity(G) # not printing because it's a huge dict of all node pair simiarities
        simrank_similarities.append(sim)
        
        if prints:
            print(f"---For model idx {i}---")
            print(f"Barycenter: {curr_barycenter}")
    
    return barycenters, simrank_similarities


def get_graph_metrics_external(mod_list, prints=False):
    
    isomorphism_checks = []; geds = []
    graphs = [make_graph(mu.get_adj_mat(mod_list[i])) for i in range(len(mod_list))]
    
    # Isomorphism check and graph edit distance
    for i in range(len(mod_list)):
        for j in range(i+1, len(mod_list)):
            G1 = graphs[i]
            G2 = graphs[j]
            
            is_isomorphic = not check_not_isomorphism(G1, G2)
            isomorphism_checks.append(is_isomorphic)
            if prints:
                print(f"---Graphs for model {i} and model {j}---")
                print(f"Is isomorphic: {is_isomorphic}")
            
            # if graphs are not isomorphic, get approximation of their edit distance
            if is_isomorphic == False:
                approx_ged = next(nx.optimize_graph_edit_distance(G1, G2))
                geds.append(approx_ged)
                if prints:
                    print(f"GED (approx): {approx_ged}")
    return isomorphism_checks, geds

def get_sorted_metrics(metric_dict, node_labels, ascending=True):
    
    # this function seems overcomplicated for sorting the count values and returning the original 
    # electrode labels but duplicates fuck up everything in mapping the labels back
    # feel free to better rewrite this function
    
    def match(x, lst):
    
        return [i for i in range(len(lst)) if lst[i] == x]

    all_counts = vu.dict_to_counts(metric_dict)

    if ascending:
        metric_sorted = sorted(all_counts)
    else:
        metric_sorted = sorted(all_counts, reverse=True)

    idx_sorted = [match(x, all_counts) for x in metric_sorted]
    
    # duplicates case
    if not all([len(x) == 1 for x in idx_sorted]):
        idx_sorted_new = []
        curr_idxs = dict()
        for i in range(len(idx_sorted)):
            if len(idx_sorted[i]) == 1:
                idx_sorted_new.extend(idx_sorted[i])
            else:
                if all_counts[idx_sorted[i][0]] not in curr_idxs.keys():
                    curr_idxs[all_counts[idx_sorted[i][0]]] = 0
                idx_sorted_new.append(idx_sorted[i][curr_idxs[all_counts[idx_sorted[i][0]]]])
                curr_idxs[all_counts[idx_sorted[i][0]]] += 1  
        idx_sorted = idx_sorted_new
    else:
        idx_sorted_new = []
        [idx_sorted_new.extend(x) for x in idx_sorted]
        idx_sorted = idx_sorted_new

    # it is possible to not have all labels if there's 2 electrodes
    # with the same count because index takes the first matching index
    assert set(idx_sorted) == set(range(22))

    labels_sorted = [node_labels[x] for x in idx_sorted]
    
    # check counts and labels in bary_sorted match original counts and labels
    assert all([all_counts[idx_sorted[i]] == metric_sorted[i] for i in range(22)])
    
    return metric_sorted, labels_sorted