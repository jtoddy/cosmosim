import numpy as np
try:
    import cupy as cp
except:
    import numpy as cp
import math

# Functions
def pairwise_distances(positions):
    n = len(positions)
    distance = cp.empty((n,n))
    for i in range(n):
        distance[i,:] = cp.abs(cp.broadcast_to(positions[i,:], positions.shape) - positions).sum(axis=1)
    return distance
    
        
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def to_cartesian(r, theta, origin=[0,0]):
    y = r*np.sin(theta) + origin[0]
    x = r*np.cos(theta) + origin[1]
    return np.array([y,x])

def to_cartesian_3d(r, theta, phi, origin=[0,0,0]):
    y = r*np.sin(theta)*np.sin(phi) + origin[0]
    x = r*np.cos(theta)*np.sin(phi) + origin[1]
    z = r*np.cos(phi) + origin[2]
    return [x,y,z]

def rotation(v,theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.matmul(R,v)

def rotation_3d(v, theta, phi):
    Rtheta = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0,1,0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    Rphi = np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])
    R = np.matmul(Rtheta,Rphi)
    return np.matmul(R,v)

def rotation_3d_multi(M, theta, phi):
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

def screen_coordinates(p, scale, offset, origin):
    return origin + (np.multiply(p,np.array([1,-1]))*scale)+(offset*scale)

def screen_coordinates_3d(p, scale, offset, rotation, origin):
    theta, phi = rotation
    v0 = rotation_3d(p, theta, phi)
    z = v0[2]
    v1 = np.delete(v0, 2, 0)
    v = origin + (np.multiply(v1,np.array([1,-1]))*scale)+(offset*scale)
    return v, z

def screen_coordinates_3d_multi(P, scale, offset, rotation, origin):
    theta, phi = rotation
    S0 = rotation_3d_multi(P, theta, phi)
    Z = S0[:,2]
    S1 = S0[:,[0,1]]
    S2 = np.multiply(S1,np.array([1,-1]))*scale
    S = S2+np.array(origin+(offset*scale))
    return S, Z