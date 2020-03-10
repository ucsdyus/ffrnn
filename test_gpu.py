import torch
# import torch before importing our library (torch runtime required)    
import ffrnn
import ffrnn_test

N = 10
R = 0.3
points = torch.rand((N, 3))

print("Point")
print(points)

print("Diag Exclude")
cnt = 0
for i in range(10):
    for j in range(10):
        if i != j:
            x = points[i]
            y = points[j]
            r = y - x
            val = torch.dot(r, r)
            if val <= R * R:
                cnt += 1
                print(i, j, r, val, R * R)
print("GT Neighbor Num", cnt)


# nn_offset, nn_list, nw_list, grad_nn_offset, grad_nn_list
cpu_res = ffrnn_test.bf_cpu(points, points, R, False)
print("CPU Offset")
print(cpu_res[0])
print(cpu_res[1])


gpu_points = points.cuda()
gpu_res = ffrnn.bf_gpu(gpu_points, gpu_points, R, False)
print("GPU Offset")
print(gpu_res[0])
