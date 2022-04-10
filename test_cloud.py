from cosmosim.core.universe import Object, Universe
#from cosmosim.core.animation import InteractiveAnimation
from cosmosim.core.animation_new import Animation
from cosmosim.util.constants import AU, ME, DE
import cosmosim.util.functions as F
import random
import math
import cProfile
import pstats
from pstats import SortKey


NUM_OBJECTS = 1000
r = 0.02*AU
omega = 2e-6

objects = []
for i in range(NUM_OBJECTS):
    m = 10*ME*random.random()
    d = DE
    p = F.random_point_in_sphere(r=r)
    v = F.rotation_3d(p*omega, math.pi/2, 0)
    obj = Object(m,d,p,velocity=v)
    objects.append(obj)
   
path = "test_data/run1b/data/"
iterations = 3000
dt = 60
collisions = True
scale=1e-7
observer_position = [0.0, 0.0, 0.1*AU]
observer_params = {"position":observer_position, "theta":0.0, "phi":0.0}

# test_sim = Universe(objects, iterations, dt=dt, outpath=path, filesize=3000)
# test_sim.run(collisions=collisions, gpu=False)

animation = Animation(path, scale=scale, observer_params=observer_params)
animation.play()
