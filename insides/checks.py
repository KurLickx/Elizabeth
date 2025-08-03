import torch

path = 'models/elis_best.pt'
state_dict = torch.load(path, map_location='cpu')  
print(state_dict.keys())
print(state_dict['lstm.weight_ih_l0'].shape)
print(state_dict['lstm.weight_ih_l0'])

best_res1v1000 = [2.4741, "00:03:51"]
best_res1v10 = [2.4784, "00:02:10"]

best_res32v1000 = [1.5754, "00:03:06" "nosence"]
best_res32v10 =[1.1869, "00:03:02" "1.4510v1time"]

best_res512v1000 = [0.1817, "00:20:10" "1.7516v32time"]
best_res512v10 = [0.1819, "00:19:34" "1.7368v32time"]           