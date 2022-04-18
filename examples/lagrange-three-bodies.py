import sys
import os
sys.path.insert(0, os.getcwd())

from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import Animation
import cosmosim.util.functions as F
import math
import numpy as np

# =============================================================================
# Lagrange's solution to the 3-body problem.
# =============================================================================

G = 1
M = 1
r = 1
pi = math.pi
omega = math.sqrt((G*M)/(math.sqrt(3)*r**3))
angles = [0.0, 2*pi/3, 4*pi/3]

objects = []
for theta in angles:
    p = F.to_cartesian_3d(r, theta, pi/2)
    v = F.rotation_3d(omega*np.array(p), pi/2, 0)
    obj = Object(mass=M, density=1000, position=p, velocity=v)
    objects.append(obj)

path = "test_data/run_lagrange/data/"
iterations = 6000
dt = 0.005
collisions = True
observer_position = [0.0, 0.0, 3]
observer_params = {"position":observer_position, "theta":0.0, "phi":0.0}

sim = Universe(objects, iterations, dt=dt, outpath=path, G=G)
sim.run(collisions=collisions, gpu=False)

animation = Animation(path, observer_params=observer_params)
animation.play(paused=True, track_all=True, obj_history_length=250)