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
from cosmosim.util.constants import ME, G as _G, C
from cosmosim.util.json_zip import json_zip
import cosmosim.util.pronounceable.main as prnc
from sklearn.metrics import pairwise_distances
import json

class Object:
    def __init__(self, mass, density, position, velocity=[0,0,0], 
                 name=None, color=None):
        self.exists = True
        self.mass = mass
        self.density = density
        self.position = np.array(position, dtype="float32")
        self.velocity = np.array(velocity, dtype="float32")
        self.name = name or (prnc.generate_word() + f"-{random.randint(1,1000)}")
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
        if gpu:
            self.xp = cp
        else:
            self.xp = np
        self.n_objects = len(objects)
        self.metadata = {}
        self.iterations = 0
        self.dt = dt
        self.gpu = gpu
        self.names = []
        self.colors = []
        self.masses = []
        self.positions = []
        self.velocities = []
        self.densities = []
        for o in objects:
            self.masses.append(o.mass)
            self.positions.append(o.position)
            self.velocities.append(o.velocity)
            self.densities.append(o.density)
            self.names.append(o.name)
            self.colors.append(o.color)
        self.masses = self.xp.array(self.masses, dtype="float32")
        self.positions = self.xp.array(self.positions, dtype="float32")
        self.velocities = self.xp.array(self.velocities, dtype="float32")
        self.densities = self.xp.array(self.densities, dtype="float32")

    def get_volumes(self):
        return self.masses/self.densities
    
    def get_radii(self):
        return ((3*self.get_volumes())/(4*math.pi))**(1/3)
        
    def get_acc(self, G):
        if self.gpu:
            return self.get_acc_gpu(G)
        else:
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

    def add_objects(self, objects):
        masses = []
        positions = []
        velocities = []
        densities = []
        for o in objects:
            self.names.append(o.name)
            self.colors.append(o.color)
            masses.append(o.mass)
            positions.append(o.position)
            velocities.append(o.velocity)
            densities.append(o.density)
        self.masses = self.xp.append(self.masses, masses)
        self.positions = self.xp.append(self.positions, positions, axis=0)
        self.velocities = self.xp.append(self.velocities, velocities, axis=0)
        self.densities = self.xp.append(self.densities, densities)
        self.n_objects = len(self.masses)

    def remove_objects(self, objects):
        remove = [self.names.index(o) for o in objects]
        keep = self.masses.astype(bool)
        keep[remove] = False
        self.densities = self.densities[keep]
        self.positions = self.positions[keep]
        self.velocities = self.velocities[keep]
        for i in sorted(remove, reverse=True):
            del self.names[i]
            del self.colors[i]
        self.n_objects = len(self.masses)
    
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
        if self.gpu:
            return self.resolve_collisions_gpu()
        else:
            n = self.n_objects
            d = pairwise_distances(self.positions, n_jobs=-1, force_all_finite=True)
            radii = self.get_radii()
            collision_matrix = d <= np.add.outer(radii,radii)
            np.fill_diagonal(collision_matrix,False)
            # Return false if no collisions
            if not np.max(collision_matrix):
                return False
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
            return True
           
    def resolve_collisions_gpu(self):
        # Initialize constants
        radii = self.get_radii()
        masses = self.masses/ME #Scale down for float32
        nan_array = cp.array([cp.nan,cp.nan,cp.nan], dtype="float32")
        # Collision matrix
        d = F.pairwise_distances(self.positions)
        collision_matrix = (d <= F.outer_sum(radii,radii))
        cp.fill_diagonal(collision_matrix, False)
        # Return false if no collisions
        if not cp.max(collision_matrix):
            return False
        # Bigger masses absorb smaller masses
        m_less = F.less(self.masses, self.masses)
        # If masses are equal, lower index absorbs higher
        m_equal = cp.tril(F.equal(self.masses, self.masses),-1)
        # Absorbtion matrix
        absorbed_matrix_leq = (m_less+m_equal)*collision_matrix
        # Each object can only be absorbed once; choose the first absorber
        absorbed_matrix_leq_first = absorbed_matrix_leq.cumsum(axis=1).cumsum(axis=1) == 1
        # Array of all objects absorbing something
        absorbing = cp.max(absorbed_matrix_leq_first, axis=0)
        # Can only be absorbed if you are not also absorbing
        absorbed_matrix = (absorbed_matrix_leq_first.T*(~absorbing)).T
        absorbed = cp.max(absorbed_matrix, axis=1)
        # Create absorbing matrix and array
        absorbing_matrix = absorbed_matrix.T
        absorbing = cp.max(absorbing_matrix, axis=1)
        # Masks for 1-D and 3-D arrays
        mask = ~absorbing_matrix
        mask_3d = cp.repeat(cp.expand_dims(mask, axis=2), 3, axis=2)
        # Calulate weighted quantities
        moments = masses[:, None] * self.positions        
        momenta = masses[:, None] * self.velocities
        densities_w = masses * self.densities
        m_totals = F.outer_sum(masses, masses)
        # Get new positions
        new_positions = F.outer_sum(moments, moments)/m_totals[None,:].T
        cp.putmask(new_positions, mask_3d, nan_array)
        new_positions = cp.nanmean(new_positions, axis=1)
        # Get new velocities
        new_velocities = F.outer_sum(momenta, momenta)/m_totals[None,:].T
        cp.putmask(new_velocities, mask_3d, nan_array)
        new_velocities = cp.nanmean(new_velocities, axis=1)
        # Get new densities
        new_densities = F.outer_sum(densities_w, densities_w)/m_totals
        cp.putmask(new_densities, mask, cp.nan)
        new_densities = cp.nanmean(new_densities, axis=1)
        # Update masses, densities, positions, and velocities
        self.masses = ME*(masses + (cp.sum(masses*absorbing_matrix, axis=1)))
        self.densities[absorbing] = new_densities[absorbing]
        self.positions[absorbing] = new_positions[absorbing]
        self.velocities[absorbing] = new_velocities[absorbing]
        # Remove absorbed objects
        self.masses = self.masses[~absorbed]
        self.densities = self.densities[~absorbed]
        self.positions = self.positions[~absorbed]
        self.velocities = self.velocities[~absorbed]
        for i in sorted(absorbed.nonzero()[0].tolist(), reverse=True):
            del self.names[i]
            del self.colors[i]
        self.n_objects = len(self.masses)
        return True
        
    def interact(self, collisions=True, G=_G):
        a = self.xp.minimum(self.get_acc(G), C)
        self.velocities = self.xp.minimum(self.velocities + a*self.dt, C)
        self.positions = self.positions + self.velocities*self.dt
        if collisions:
            colliding = True
            while colliding:
                colliding = self.resolve_collisions()
        self.iterations += 1
        
    def state_json(self):
        return {"n": self.n_objects,
                "dt": self.dt,
                "names": self.names.copy(),
                "colors": self.colors.copy(),
                "masses": cp.asarray(self.masses).tolist(),
                "densities": cp.asarray(self.densities).tolist(),
                "positions": cp.asarray(self.positions).tolist(),
                "velocities": cp.asarray(self.velocities).tolist(),
                "iterations": self.iterations}
        
class Universe:
    
    def __init__(self, objects, iterations, dt=1, G=_G, outpath=None, filesize=None):
        self.objects = objects
        self.dt = dt
        self.iterations = iterations
        self.outpath = outpath
        self.filesize = filesize
        self.G = G
        
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
            if np.sum(m) == 0:
                m_valid = False
                problems.append("All entries are 0")
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
                    state.interact(collisions, G=self.G)
                    state_json = state.state_json()
                    if validate:
                        valid, invalid_matrices = self.validate_state(state_json)
                        if not valid:
                            print("\nInvalid matrices:\n"+json.dumps(invalid_matrices,indent=2))
                            raise(Exception("Data is invalid"))
                    states.append(state_json)
                    elapsed += 1
                    if elapsed >= self.iterations: 
                        break                    
                print(f"Writing file {n+1}/{nfiles}...")
                with open(path, "w") as f:
                    #json.dump(states,f)
                    states_compressed = json_zip(states)
                    json.dump(states_compressed, f)
                print("Done!")
        else:
            states = []
            for i in tqdm(range(self.iterations), desc="Running simulation"):
                state.interact(collisions)
                states.append(state.state_json())
            return states
            
    
    