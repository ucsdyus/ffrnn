import torch
# import torch before importing our library (torch runtime required)    
import ffrnn


points = torch.rand((10, 3))
R = 0.3

nn_list = ffrnn.bf_cpu(points, R)

print("Point")
print(points)

print("SelectMat")
for i, (neighbor, weight) in enumerate(nn_list):
    print("ID:", i)
    print(neighbor.size(), neighbor)
    print(weight.size(), weight.sum(axis=1), weight)