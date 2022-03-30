import numpy as np
import random
import math
import pickle
import os
import copy
from tqdm import tqdm
import cosmosim.util.functions as F
from cosmosim.util.blas import acc_blas
from sklearn.metrics import pairwise_distances
import cosmosim.util.pronounceable.main as prnc

AU = 1.496e11       # Astronomical unit
ME = 5.972e24       # Mass of the Earth
RE = 6.371e6        # Radius of the earth
MS = 1.989e30       # Mass of the sun
RS = 6.9634e8       # Radius of the sun
DAYTIME = 86400     # Seconds in a day
_G = 6.674e-11      # Gravitational constant
C = 3e8             # Speed of light

class Object:
    def __init__(self, mass, density, position, velocity=[0,0,0], 
                 name=None, color=None):
        self.exists = True
        self.mass = mass
        self.density = density
        self.position = np.array(position).astype(float)
        self.velocity = np.array(velocity).astype(float)
        self.name = name or prnc.generate_word()
        self.color = color or (int(255*random.random()),int(255*random.random()),int(255*random.random()))
        
    def get_volume(self):
        return self.mass/self.density
    
    def get_radius(self):
        return ((3*self.get_volume())/(4*math.pi))**(1/3)
        
    def create_satellite(self, distance=None, mass=None, density=None, 
                         theta=None, name=None, color=None, G=_G):
        distance = distance or random.randint(int(self.get_radius()*5), int(self.get_radius()*100))
        mass = mass or random.random()*self.mass
        density = density or self.density
        theta = theta or 2*math.pi*random.random()
        v_mag = math.sqrt((self.mass*G)/distance) # Circular orbit
        pos = F.to_cartesian(distance, theta)
        pos_norm = pos/np.linalg.norm(pos)
        v = np.append(F.rotation(v_mag*pos_norm,math.pi/2), [0])
        pos = np.append(pos, [0])
        obj = Object(mass, density, pos, v, name, color)
        return obj


class State:
    def __init__(self, objects, dt=1):
        self.n_objects = len(objects)
        self.masses = []
        self.positions = []
        self.velocities = []
        self.densities = []
        self.names = []
        self.colors = []
        self.iterations = 0
        self.dt = dt
        for o in objects:
            self.masses.append(o.mass)
            self.positions.append(o.position)
            self.velocities.append(o.velocity)
            self.densities.append(o.density)
            self.names.append(o.name)
            self.colors.append(o.color)
        self.masses = np.array(self.masses).astype(float)
        self.positions = np.array(self.positions).astype(float)
        self.velocities = np.array(self.velocities).astype(float)
        self.densities = np.array(self.densities).astype(float)

    def get_volumes(self):
        return self.masses/self.densities
    
    def get_radii(self):
        return ((3*self.get_volumes())/(4*math.pi))**(1/3)

    def collide(self, i, j):
        m_i = self.masses[i]
        m_j = self.masses[j]
        p_i = self.positions[i]
        p_j = self.positions[j]
        v_i = self.velocities[i]
        v_j = self.velocities[j]
        d_i = self.densities[i]
        d_j = self.densities[j]
        m_total = max(m_i+m_j, 1.0)
        p = ((m_i*p_i)+(m_j*p_j))/m_total
        v = ((m_i*v_i)+(m_j*v_j))/m_total
        d = ((m_i*d_i)+(m_j*d_j))/m_total
        self.masses[i] = m_total
        self.masses[j] = 0.0
        self.positions[i] = p
        self.velocities[i] = v
        self.velocities[j] = 0.0
        self.densities[i] = d
        
    def get_acc(self, G):
        return acc_blas(self.positions, self.masses, G)  # Magic!!!
        # xp = np
        # mass_matrix = self.masses.reshape((1, -1, 1))*self.masses.reshape((-1, 1, 1))
        # disps = self.positions.reshape((1, -1, 3)) - self.positions.reshape((-1, 1, 3)) # displacements
        # dists = xp.linalg.norm(disps, axis=2)
        # dists[dists == 0] = 1 # Avoid divide by zero warnings
        # forces = G*disps*mass_matrix/xp.expand_dims(dists, 2)**3
        # return forces.sum(axis=1)/self.masses.reshape(-1, 1)
    
    def resolve_collisions(self):
        d = pairwise_distances(self.positions, n_jobs=-1, force_all_finite=True)
        radii = self.get_radii()
        collision_matrix = d <= np.add.outer(radii,radii)
        absorbed = []
        for i in range(self.n_objects):
            if i not in absorbed:
                for j, c in enumerate(collision_matrix[i]):
                        if c and i != j and j not in absorbed:
                            if self.masses[i] >= self.masses[j]:
                                self.collide(i,j)
                                absorbed.append(j)
                            else:
                                self.collide(j,i)
                                absorbed.append(i)
        self.masses = np.delete(self.masses, absorbed)
        self.positions = np.delete(self.positions, absorbed, axis=0)
        self.velocities = np.delete(self.velocities, absorbed, axis=0)
        self.densities = np.delete(self.densities, absorbed)
        for i in sorted(absorbed, reverse=True):
            try:
                del self.names[i]
                del self.colors[i]
            except Exception as e:
                print(i)
                print(self.n_objects)
                print(collision_matrix.shape)
                raise(e)
        self.n_objects -= len(absorbed)
            
    def interact(self, collisions=True, G=_G):
        a = self.get_acc(G)
        self.velocities = np.minimum(self.velocities + a*self.dt, C)
        self.positions = self.positions + self.velocities*self.dt
        if collisions:
            self.resolve_collisions()
        self.iterations += 1

    def save(self, f):
        pickle.dump(self, f)
        
        
class Universe:
    
    def __init__(self, objects, dt, iterations, outpath=None, filesize=1000):
        self.objects = objects
        self.dt = dt
        self.iterations = iterations
        self.outpath = outpath
        self.filesize = filesize
        print(outpath)
               
    def run(self, collisions=True):
        state = State(self.objects, dt=self.dt)
        nfiles = math.ceil(self.iterations/self.filesize)
        elapsed = 0
        if self.outpath:
            if not os.path.isdir(self.outpath):
                os.makedirs(self.outpath)
            existing_filelist = os.listdir(self.outpath)
            for f in existing_filelist:
                os.remove(self.outpath + f)
            print(f"A total of {nfiles} data files will be created.")
            for n in range(nfiles): 
                path = self.outpath + f"{n}.dat"
                for i in tqdm(range(min(self.filesize, self.iterations)), desc=f"Writing file {n}"):
                    state.interact(collisions)
                    with open(path, "ab+") as f:
                        state.save(f)
                    elapsed += 1
                    if elapsed >= self.iterations:
                        break
        else:
            states = []
            for i in tqdm(range(self.iterations), desc="Running simulation"):
                state.interact()
                new_state = copy.deepcopy(state)
                states.append(new_state)
            return states
            
    
    