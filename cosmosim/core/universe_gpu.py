import numpy as np
import cupy as cp
import random
import math
import pickle
import os
import copy
from tqdm import tqdm
import cosmosim.util.functions_gpu as F
from cosmosim.util.blas import acc_blas
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
        
        

class Snapshot:
    def __init__(self, masses, positions, velocities, radii, colors):
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.radii = radii
        self.colors = colors
        self.iterations = 0
        
    def get_acc(self, G):
        xp = cp.get_array_module(self.positions)
        mass_matrix = self.masses.reshape((1, -1, 1))*self.masses.reshape((-1, 1, 1))
        disps = self.positions.reshape((1, -1, 3)) - self.positions.reshape((-1, 1, 3)) # displacements
        dists = xp.linalg.norm(disps, axis=2)
        dists[dists == 0] = 1 # Avoid divide by zero warnings
        forces = G*disps*mass_matrix/xp.expand_dims(dists, 2)**3
        return forces.sum(axis=1)/self.masses.reshape(-1, 1)
    
    def resolve_collisions(self):
        if self.collisions:
            d = F.pairwise_distances(self.positions).get()
            collision_matrix = d <= np.add.outer(self.radii,self.radii)
            for i in range():
                pass
            
    def interact(self, collisions=True, G=_G, dt=1):
        a = self.get_acc(G)
        self.velocities += a*dt
        self.positions += self.velocities*dt
        self.resolve_collisions()
        self.iterations += 1
            
            
            

class State:
    
    def __init__(self, objects, dt=1, G=_G, iteration=0):
        self.objects = objects
        self.dt = dt
        self.G = G
        self.iteration = iteration
        
    def masses(self):
        return cp.array([o.mass for o in self.objects], dtype='float32')
    
    def radii(self):
        return np.array([o.get_radius() for o in self.objects], dtype='float32')
    
    def positions(self):
        return cp.array([o.position for o in self.objects], dtype='float32')
    
    def velocities(self):
        return cp.array([o.velocity for o in self.objects], dtype='float32')
    
    def colors(self):
        return [o.color for o in self.objects]
    
    def interact(self, collisions=True):
        m = self.masses()
        r = self.radii()
        v0 = self.velocities()
        p0 = self.positions()
        
        # Calculate net accelerations
        #a = acc_blas(p0, m, self.G)  # Magic!!!
        a = F.get_acc(p0, m, self.G)
        # Integration
        v = v0 + a*self.dt
        p = p0 + v*self.dt
        
        if collisions:
            d = F.pairwise_distances(p).get()
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
               
    def run(self, collisions=True):
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
            
    
    