import torch
# import torch before importing our library (torch runtime required)    
import ffrnn
import ffrnn_test

N = 10
R = 0.3
points = torch.tensor([[0.4862, 0.2957, 0.3388],
        [0.9290, 0.7391, 0.1515],
        [0.1459, 0.5053, 0.4150],
        [0.5393, 0.8685, 0.8356],
        [0.9538, 0.3375, 0.1379],
        [0.8930, 0.4342, 0.8626],
        [0.3683, 0.7003, 0.8116],
        [0.4727, 0.7904, 0.6609],
        [0.3455, 0.4732, 0.8694],
        [0.6546, 0.5523, 0.5124]], dtype=torch.float32)

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
print(gpu_res[1])
print(gpu_res[3])
print(gpu_res[4])
