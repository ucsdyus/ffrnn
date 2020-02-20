import torch
import numpy as np 
# cuda0 = torch.device('cuda:0')

def ball2cyl(points): # defined as x,y,z row vector
  cyl_points = torch.zeros(points.size(), dtype=points.dtype, device=points.device)
  x = points[:,0]
  y = points[:,1]
  z = points[:,2]
  r = torch.norm(points, dim=1)
  cond1 = (r == 0)
  # print(cond1)
  cond2 = (x**2 + y**2 >= (5/4 * z**2))
  cond3 = ~(cond1 | cond2)
  
  coeff2 = r[cond2] / torch.norm(points[cond2][:,:2], dim=1)
  cyl_points[cond2] = torch.stack((x[cond2] * coeff2, \
    y[cond2] * coeff2, 1.5*z[cond2]), 1)
  
  coeff3 = torch.sqrt(3*r[cond3] / (r[cond3] + torch.abs(z[cond3])))
  cyl_points[cond3] = torch.stack((x[cond3]*coeff3, \
    y[cond3]*coeff3, torch.sign(z[cond3])), 1)

  # print("ball2cycl cond1", cond1)
  # print("ball2cycl cond2", cond2)
  # print("ball2cycl cond3", cond3)
  return cyl_points

def cyl2cube(points): 
  x = points[:,0]
  y = points[:,1]
  z = points[:,2]
  # print("cyl2cube", points)
  xy_norm = torch.norm(points[:,:2], dim=1)
  cond1 = (xy_norm == 0)
  cond2 = (torch.abs(y) <= torch.abs(x))
  cond3 = ~(cond1 | cond2)
  cube_points = torch.zeros(points.size(), dtype=points.dtype, device=points.device)
  cube_points[cond1][:,2] = z[cond1]

  cube_points[cond2] = torch.stack((xy_norm[cond2] * np.sign(x[cond2]), \
    4/np.pi * np.sign(x[cond2]) * xy_norm[cond2] * np.arctan(y[cond2] / x[cond2]), \
    z[cond2]),1)

  cube_points[cond3] = torch.stack((4/np.pi * torch.sign(y[cond3])*xy_norm[cond3] \
    * torch.atan(x[cond3] / y[cond3]), \
    xy_norm[cond3] * torch.sign(y[cond3]), z[cond3]),1)

  # print("cyl2cube cond1", cond1)
  # print("cyl2cube cond2", cond2)
  # print("cyl2cube cond3", cond3)
  
  return cube_points


def sp2rect(points): 
  return 0.5*cyl2cube(ball2cyl(points)) + torch.tensor([0.5,0.5,0.5])

# T = 10
# x = torch.rand(T, 3)
# x /= torch.norm(x,dim=1).view(T, 1)
# a = sp2rect(x)
# print(a.data.numpy())