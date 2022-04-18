import numpy as np
import random
try:
    import cupy as cp
except:
    import numpy as cp
import math

# Functions
def pairwise_displacements(positions):
    disps = positions.reshape((1, -1, 3)) - positions.reshape((-1, 1, 3))
    return disps

def pairwise_distances(positions):
    disps = pairwise_displacements(positions)
    dists = cp.linalg.norm(disps, axis=2)
    return dists
        
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def cartesian_distance(p1, p2):
    arg = 0
    for i in range(len(p1)):
        arg += (p1[i]-p2[i])**2
    return math.sqrt(arg)

def random_point_in_sphere(r=1, origin=[0.0,0.0,0.0]):
    u = random.random()
    x = np.random.normal()
    y = np.random.normal()
    z = np.random.normal()
    mag = (x**2 + y**2 + z**2)**(1/2)
    x /= mag
    y /= mag
    z /= mag
    c = r*u**(1/3)
    return np.array([x*c, y*c, z*c]) + np.array(origin)

def to_cartesian(r, theta, origin=[0,0]):
    y = r*np.sin(theta) + origin[0]
    x = r*np.cos(theta) + origin[1]
    return np.array([y,x])

def to_cartesian_3d(r, theta, phi, origin=[0,0,0]):
    y = r*np.sin(theta)*np.sin(phi) + origin[0]
    x = r*np.cos(theta)*np.sin(phi) + origin[1]
    z = r*np.cos(phi) + origin[2]
    return [x,y,z]

def Rx(theta):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta), np.cos(theta)]
    ])
    return Rx

def Ry(theta):
    Ry = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)]
    ])
    return Ry
    
def Rz(theta):
    Rz = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
        
    ])
    return Rz

def rotation(v,theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.matmul(R,v)

def rotation_3d(v, theta, phi):
    Rtheta = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
        
    ])
    Rphi = np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])
    R = np.matmul(Rtheta,Rphi)
    return np.matmul(R,v)

def rotation_3d_multi(M, theta, phi):
    M = np.array(M)
    Rtheta = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0,1,0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    Rphi = np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])
    R = np.matmul(Rtheta,Rphi)
    return R.dot(M.T).T

def get_tangent(position, reference=[0,0]):
    R = np.array([[0,1],[-1,0]])
    dp = reference - position
    tangent = R @ normalize(dp)
    return tangent

def get_radius(mass, density):
    volume = mass/density
    radius = ((3*volume)/(4*math.pi))**(1/3)
    return radius

def outer_sum(a, b):
    return a[:, None] + b[None, :]

def less_equal(a, b):
    return (a[:, None] <= b[None, :])

def less(a, b):
    return (a[:, None] < b[None, :])

def equal(a, b):
    return (a[:, None] == b[None, :])