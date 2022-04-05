from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import InteractiveAnimation
from cosmosim.util.constants import AU, ME, DE
import cosmosim.util.functions as F
import numpy as np
import random
import math


omega = 1e-6 # angular velocity

# Create some planets in a disk around the star
NUM_OBJECTS = 500
objects = []
for i in range(NUM_OBJECTS):
    m = 10*ME*random.random()
    d = DE
    r = 0.1*AU*random.random()
    theta = 2*math.pi*random.random()
    phi = math.pi*random.random()
    p = np.array(F.to_cartesian_3d(r, theta, phi))
    v = F.rotation_3d(p*omega, math.pi/2, 0)
    obj = Object(m,d,p,velocity=v)
    objects.append(obj)
   
#Simulate
path = "test_data/run1b/data/"
iterations = 1000
dt = 600
collisions = True
scale=6.5e-9

# test_sim = Universe(objects, iterations, dt=dt, outpath=path, filesize=3000)
# test_sim.run(collisions=collisions, gpu=False)
animation = InteractiveAnimation(path, scale=scale)
animation.play()