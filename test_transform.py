import torch
# import torch before importing our library (torch runtime required)    
import ffrnn
from toRect_torch import sp2rect

T = 2
v = torch.rand(T, 3, dtype=torch.float32)
v /= (torch.norm(v, dim=1).view(T, 1) + 1e-5)
x = torch.stack([v, -v], 0).view(-1, 3)
T *= 2
# T = 1
# x = torch.tensor([-0.1481, -0.8331, -0.5330], dtype=torch.float32).view(1, 3)
R = 2.0


h_gt = sp2rect(x)

h = torch.zeros((len(x), 3), dtype=torch.float32)
g = torch.zeros((len(x), 4 * 4 * 4), dtype=torch.float32)
g_win = torch.zeros((len(x), 4 * 4 * 4), dtype=torch.float32)
for i in range(T):
    print("|r|", i, "=", torch.dot(x[i], x[i]), x[i])
    h[i] = ffrnn.th_ball2cube(x[i])
    g[i] = ffrnn.th_ball2grid(x[i])
    g_win[i] = ffrnn.th_ball2grid_with_window(x[i])
    print(torch.norm(h_gt[i] - h[i]))

print("Ball")
print(x)
print("Cube")
print(h)
print("Cube GT")
print(h_gt)

print("Grid")
print(torch.sum(g, axis=1))

print("Grid with window")
print(torch.sum(g_win, axis=1))

v = 1.0 - (1.0 / R) ** 2
print("window weights:", v * v * v)
