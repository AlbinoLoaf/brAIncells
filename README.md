# DGCNN graph stability inference 
**wit Pytorch, NetworkX, torcheeg**

This is the code base for our bachelor thesis, in which we investigate the observed phenomenon of unstable learned graph produced by the DGCNN. 

## Usage
Most of the functionality in this project is found in the five utils files listed below. We have endevoured to keep the code as modular as posible making it easy to insert new modules. To run all our experiments and get the same results as us train modes with 8,16,24,32,40,48,56,64,128 hidden channels for the seeds listed in seeds.txt and run model_enviroment.ipynb.

To run the experiments with new models there is a script version of the code run_many_models,py. change PARAM_LIST variable to the desired hidden channels and setnum_seeds to the desired number of seed. Run the script on a powerfull machine or cluster and then run the model_enviroment.ipynb

### Util files
- cka.py
- data_utils.py
- graph_utils.py
- model_utils.py
- visual_utils.py


## Setup 
We are working in a conda inviroment with python 3.10, with the "requirements.txt" file describing package requirements. 


## Authors

| Author name           | author alias |
| --------------------- | ------------ |
| Cristina "Keli" Avram | keli1111     |
| Lili Y. Raleva        | liliraleva   |
| Marius Thomsen        | AlbinoLoaf   |

Supervised by Laurits Dixen