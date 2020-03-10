import torch
# import torch before importing our library (torch runtime required)    
import ffrnn


points = torch.rand((10, 3))
R = 0.3
print("Point")
print(points)

print("Diag Exclude")
nn_offset, nn_list, nw_list, grad_nn_offset, grad_nn_list = ffrnn.bf_cpu(points, points, R, False)

print("nn_offset", nn_offset)
print("grad_nn_offset", nn_offset)

print("SelectMat")
for i in range(len(nn_offset) - 1):
    print("ID:", i)
    start = nn_offset[i]
    end = nn_offset[i + 1]
    Ns = end - start
    print(Ns, nn_list[start:end])

print("Diag include")
nn_offset, nn_list, nw_list, grad_nn_offset, grad_nn_list = ffrnn.bf_cpu(points, points, R, True)

print("nn_offset", nn_offset)
print("grad_nn_offset", nn_offset)

print("SelectMat")
for i in range(len(nn_offset) - 1):
    print("ID:", i)
    start = nn_offset[i]
    end = nn_offset[i + 1] 
    Ns = end - start
    print(Ns, nn_list[start:end])
