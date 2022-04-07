from cosmosim.core.universe import Object, Universe
#from cosmosim.core.animation import InteractiveAnimation
from cosmosim.core.animation_new import Animation
from cosmosim.util.constants import AU, ME, DE
import cosmosim.util.functions as F
import random
import math


NUM_OBJECTS = 500
r = 0.01*AU
omega = 1e-6

objects = []
for i in range(NUM_OBJECTS):
    m = 10*ME*random.random()
    d = DE
    p = F.random_point_in_sphere(r=r)
    v = F.rotation_3d(p*omega, 0, math.pi/2)
    obj = Object(m,d,p,velocity=v)
    objects.append(obj)
   
path = "test_data/run1b/data/"
iterations = 1000
dt = 6
collisions = True
scale=1e-7

# test_sim = Universe(objects, iterations, dt=dt, outpath=path, filesize=3000)
# test_sim.run(collisions=collisions, gpu=False)
animation = Animation(path, scale=scale)
animation.play()