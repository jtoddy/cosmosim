from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import InteractiveAnimation
from cosmosim.util.constants import AU, ME, DE
import cosmosim.util.functions as F
import random
import math




# Create some planets in a disk around the star
NUM_OBJECTS = 1000
objects = []
for i in range(NUM_OBJECTS):
    m = 10*ME*random.random()
    d = DE
    r = 0.1*AU*random.random()
    theta = 2*math.pi*random.random()
    phi = math.pi*random.random()
    p = F.to_cartesian_3d(r, theta, phi)
    obj = Object(m,d,p)
    objects.append(obj)
   
#Simulate
path = "test_data/run1b/data/"
iterations = 30000
dt = 600
collisions = True
scale=6.5e-9

# test_sim = Universe(objects, iterations, dt=dt, outpath=path, filesize=3000)
# test_sim.run(collisions=collisions, gpu=True)
animation = InteractiveAnimation(path, scale=scale)
animation.play()