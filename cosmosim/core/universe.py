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
    
    def absorb(self, obj):
        # Inelastic collision
        m_total = self.mass + obj.mass
        self.position = ((self.mass*self.position)+(obj.mass*obj.position))/m_total
        self.velocity = ((self.mass*self.velocity)+(obj.mass*obj.velocity))/m_total
        self.density = ((self.mass*self.density)+(obj.mass*obj.density))/m_total
        self.mass += obj.mass
        obj.destroy()
    
    def destroy(self):
        self.exists = False
        self.mass = 0.0
        self.volume = 0.0
        self.radius = 0.0
        self.velocity = np.array([0.0,0.0,0.0])
        self.position = np.array([0.0,0.0,0.0])
        
    def create_satellite(self, distance=None, mass=None, density=None, 
                         theta=None, name=None, color=None, G=_G):
        distance = distance or random.randint(int(self.radius*5), int(self.radius*100))
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
    
    def __init__(self, objects, dt=1, G=_G, iteration=0):
        self.objects = objects
        self.dt = dt
        self.G = G
        self.iteration = iteration
        
    def masses(self):
        return np.array([o.mass for o in self.objects])
    
    def radii(self):
        return np.array([o.get_radius() for o in self.objects])
    
    def positions(self):
        return np.array([o.position for o in self.objects])
    
    def velocities(self):
        return np.array([o.velocity for o in self.objects])
    
    def colors(self):
        return [o.color for o in self.objects]
    
    def interact(self,  collisions=True):
        m = self.masses()
        r = self.radii()
        v0 = self.velocities()
        p0 = self.positions()
        
        # Calculate net accelerations
        a = acc_blas(p0, m, self.G)  # Magic!!!
        # Integration
        v = v0 + a*self.dt
        p = p0 + v*self.dt
        
        if collisions:
            d = pairwise_distances(p, n_jobs=-1, force_all_finite=True)
            collision_matrix = d <= np.add.outer(r,r)
        
        # Update objects
        absorbed = []
        for i, obj in enumerate(self.objects):
            if obj.exists:
                obj.velocity = v[i]
                obj.position = p[i]
                if collisions:
                    for j, c in enumerate(collision_matrix[i]):
                        if c and i != j:
                            other_obj = self.objects[j]
                            if obj.mass >= other_obj.mass:
                                obj.absorb(other_obj)
                                absorbed.append(other_obj)
        for obj in absorbed:
            if obj in self.objects:
                self.objects.remove(obj)
        self.iteration += 1
        
    def save(self, f):
        pickle.dump(self, f)
        
        
class Universe:
    
    def __init__(self, objects, dt, iterations, outpath=None, filesize=1000):
        self.objects = objects
        self.dt = dt
        self.iterations = iterations
        self.outpath = outpath
        self.filesize = filesize
               
    def run(self):
        state = State(self.objects, dt=self.dt)
        nfiles = math.ceil(self.iterations/self.filesize)
        elapsed = 0
        if self.outpath:
            if not os.path.isdir(self.outpath):
                os.mkdir(self.outpath)
            existing_filelist = os.listdir(self.outpath)
            for f in existing_filelist:
                os.remove(self.outpath + f)
            print(f"A total of {nfiles} data files will be created.")
            for n in range(nfiles): 
                path = self.outpath + f"{n}.dat"
                for i in tqdm(range(min(self.filesize, self.iterations)), desc=f"Writing file {n}"):
                    state.interact(self.dt)
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
            
    
    