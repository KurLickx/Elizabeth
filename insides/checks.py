import torch

path = 'models/elis_best.pt'
state_dict = torch.load(path, map_location='cpu')  

print(state_dict.keys())
print(state_dict['lstm.weight_ih_l0'].shape)
print(state_dict['lstm.weight_ih_l0'])