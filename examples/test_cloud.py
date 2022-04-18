import sys
import os
sys.path.insert(0, os.getcwd())

from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import Animation
from cosmosim.util.constants import AU, ME, DE
import cosmosim.util.functions as F
import random
import math


NUM_OBJECTS = 1000
r = 0.02*AU
omega = 8e-6

objects = []
for i in range(NUM_OBJECTS):
    m = 10*ME*random.random()
    d = DE
    p = F.random_point_in_sphere(r=r)
    phi = abs(math.atan(p[1]/p[0]))
    v = F.rotation_3d(p*omega, math.pi/2, 0)*math.cos(phi)
    obj = Object(m,d,p,velocity=v)
    objects.append(obj)
   
path = "test_data/run1/data/"
iterations = 60000
dt = 60
collisions = True
observer_position = [0.0, 0.0, 0.1*AU]
observer_params = {"position":observer_position, "theta":0.0, "phi":0.0}

# test_sim = Universe(objects, iterations, dt=dt, outpath=path, filesize=10000)
# test_sim.run(collisions=collisions, gpu=False)

animation = Animation(path, observer_params=observer_params)
animation.play()
