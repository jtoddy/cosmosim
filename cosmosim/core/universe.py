import numpy as np
try:
    import cupy as cp
except:
    import numpy as cp
import random
import math
import os
from tqdm import tqdm
import cosmosim.util.functions as F
from cosmosim.util.blas import acc_blas
import cosmosim.util.pronounceable.main as prnc
from sklearn.metrics import pairwise_distances
import json

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
        self.position = np.array(position, dtype="float32")
        self.velocity = np.array(velocity, dtype="float32")
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
    def __init__(self, objects, dt=1, gpu=False):
        self.n_objects = len(objects)
        self.masses = []
        self.positions = []
        self.velocities = []
        self.densities = []
        self.names = []
        self.colors = []
        self.metadata = {}
        self.iterations = 0
        self.dt = dt
        self.gpu = gpu
        self.scale = ME
        for o in objects:
            self.masses.append(o.mass)
            self.positions.append(o.position)
            self.velocities.append(o.velocity)
            self.densities.append(o.density)
            self.names.append(o.name)
            self.colors.append(o.color)
        if gpu:
            self.masses = cp.array(self.masses, dtype="float32")
            self.positions = cp.array(self.positions, dtype="float32")
            self.velocities = cp.array(self.velocities, dtype="float32")
            self.densities = cp.array(self.densities, dtype="float32")
        else:
            self.masses = np.array(self.masses, dtype="float32")
            self.positions = np.array(self.positions, dtype="float32")
            self.velocities = np.array(self.velocities, dtype="float32")
            self.densities = np.array(self.densities, dtype="float32")

    def get_volumes(self):
        return self.masses/self.densities
    
    def get_radii(self):
        return ((3*self.get_volumes())/(4*math.pi))**(1/3)
        
    def get_acc(self, G):
        return acc_blas(self.positions, self.masses, G)  # Magic!!!
    
    def get_acc_gpu(self, G):
        masses = self.masses/ME
        mass_matrix = masses.reshape((1, -1, 1))*masses.reshape((-1, 1, 1))
        disps = self.positions.reshape((1, -1, 3)) - self.positions.reshape((-1, 1, 3))
        dists = cp.linalg.norm(disps, axis=2)
        dists[dists == 0] = 1
        forces = (ME*G*disps*mass_matrix)/cp.expand_dims(dists, 2)**3
        a = (forces.sum(axis=1)/masses.reshape(-1, 1))
        return a
    
    def collide(self, i, j):
        m_i = self.masses[i]/ME
        m_j = self.masses[j]/ME
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
        self.masses[i] = m_total*ME
        self.positions[i] = p
        self.velocities[i] = v
        self.densities[i] = d
    
    def resolve_collisions(self):
        n = self.n_objects
        d = pairwise_distances(self.positions, n_jobs=-1, force_all_finite=True)
        radii = self.get_radii()
        collision_matrix = d <= np.add.outer(radii,radii)
        np.fill_diagonal(collision_matrix,False)
        absorbed = []
        for i in range(n):
            if i not in absorbed:
                for j, c in enumerate(collision_matrix[i]):
                        if c and j not in absorbed:
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
            del self.names[i]
            del self.colors[i]
        self.n_objects = len(self.masses)
           
    def resolve_collisions_gpu(self):
        radii = self.get_radii()
        d = F.pairwise_distances(self.positions)
        masses = self.masses/ME

        collision_matrix = (d <= F.outer_sum(radii,radii))
        cp.fill_diagonal(collision_matrix, False)
        cm = cp.max(collision_matrix, axis=1)

        momenta = masses[:, None] * self.velocities
        moments = masses[:, None] * self.positions
        densities_w = masses * self.densities
        m_totals = F.outer_sum(masses, masses)
        
        new_positions = (moments + moments[:, None, :])/m_totals[None,:].T
        new_positions[~collision_matrix] = 0
        new_positions = cp.sum(new_positions, axis=1)
        
        new_velocities = (momenta + momenta[:, None, :])/m_totals[None,:].T
        new_velocities[~collision_matrix] = 0
        new_velocities = cp.sum(new_velocities, axis=1)
        
        new_densities = F.outer_sum(densities_w, densities_w)/m_totals
        new_densities[~collision_matrix] = None
        new_densities = cp.nanmean(new_densities, axis=1)
        
        m_less = F.less(self.masses, self.masses)
        m_equal = cp.tril(F.equal(self.masses, self.masses),-1)
        abs_matrix = (m_less+m_equal)*collision_matrix
        absorbed = cp.max(abs_matrix, axis=1)

        self.masses = (masses + cp.sum(collision_matrix*masses, axis=1))*ME
        self.densities[cm] = new_densities[cm]
        self.positions[cm] = new_positions[cm]
        self.velocities[cm] = new_velocities[cm]
        
        
        
        self.masses = self.masses[~absorbed]
        self.densities = self.densities[~absorbed]
        self.positions = self.positions[~absorbed]
        self.velocities = self.velocities[~absorbed]
        
        for i in sorted(absorbed.nonzero()[0].tolist(), reverse=True):
            del self.names[i]
            del self.colors[i]
        self.n_objects = len(self.masses)
        
    def interact(self, collisions=True, G=_G):
        if self.gpu:
            a = cp.minimum(self.get_acc_gpu(G), C)
            self.velocities = cp.minimum(self.velocities + a*self.dt, C)
        else:  
            a = np.minimum(self.get_acc(G), C)
            self.velocities = np.minimum(self.velocities + a*self.dt, C)
        self.positions = self.positions + self.velocities*self.dt
        if collisions and self.gpu:
            self.resolve_collisions_gpu()
        elif collisions:
            self.resolve_collisions()
        self.iterations += 1
        
    def state_json(self):
        return {"n": self.n_objects,
                "names": self.names.copy(),
                "colors": self.colors.copy(),
                "masses": cp.asarray(self.masses).tolist(),
                "densities": cp.asarray(self.densities).tolist(),
                "positions": cp.asarray(self.positions).tolist(),
                "velocities": cp.asarray(self.velocities).tolist(),
                "iterations": self.iterations}
        
class Universe:
    
    def __init__(self, objects, dt, iterations, outpath=None, filesize=None):
        self.objects = objects
        self.dt = dt
        self.iterations = iterations
        self.outpath = outpath
        self.filesize = filesize
        
    def validate_state(self, state):
        invalid_matrices = {}
        valid = True
        n = state["n"]
        for _m in ["positions","velocities","masses","densities"]:
            m = state[_m]
            problems = []
            m_valid = True
            if np.isnan(np.sum(m)):
                m_valid = False
                problems.append("NaN values detected")
            if len(m) != n:
                m_valid = False
                p = "too big" if len(m) > n else "too small"
                problems.append(f"Matrix size is {p}")
            if not m_valid:
                valid = False
                invalid_matrices[_m] = problems
                
        return valid, invalid_matrices
        
    def run(self, collisions=True, gpu=False, validate=True):
        if gpu:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        state = State(self.objects, dt=self.dt, gpu=gpu)
        nfiles = math.ceil(self.iterations/self.filesize) if self.filesize else 1
        elapsed = 0
        if self.outpath:
            if not os.path.isdir(self.outpath):
                os.makedirs(self.outpath)
            existing_filelist = os.listdir(self.outpath)
            for f in existing_filelist:
                os.remove(self.outpath + f)
            print(f"A total of {nfiles} data files will be created.")
            for n in range(nfiles): 
                path = self.outpath + f"data_{n}.json"
                states = []
                for i in tqdm(range(min(self.filesize or np.inf, self.iterations)), desc=f"Preparing file {n+1}/{nfiles}"):
                    state.interact(collisions)
                    state_json = state.state_json()
                    if validate:
                        valid, invalid_matrices = self.validate_state(state_json)
                        if not valid:
                            print("Invalid matrices: "+str(invalid_matrices))
                            raise(Exception("Data is invalid"))
                    states.append(state_json)
                    elapsed += 1
                    if elapsed >= self.iterations:
                        break                    
                print(f"Writing file {n}...")
                with open(path, "w") as f:
                    json.dump(states,f)
        else:
            states = []
            for i in tqdm(range(self.iterations), desc="Running simulation"):
                state.interact(collisions)
                states.append(state.state_json())
            return states
            
    
    