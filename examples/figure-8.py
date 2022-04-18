import sys
import os
sys.path.insert(0, os.getcwd())

from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import Animation

# =============================================================================
# Figure-8 solution to the 3-body problem
# https://arxiv.org/abs/1705.00527v4
# =============================================================================

p1 = [-1.0, 0.0, 0.0]
p2 = [1.0, 0.0, 0.0] 
p3 = [0.0, 0.0, 0.0]

v1 = [0.3471168881,0.5327249454, 0]
v2 = [0.3471168881,0.5327249454, 0]
v3 = [-0.6942337762,-1.0654498908, 0] 

planet1 = Object(mass=1, density=1000, position=p1, velocity=v1)
planet2 = Object(mass=1, density=1000, position=p2, velocity=v2)
planet3 = Object(mass=1, density=1000, position=p3, velocity=v3)
objects = [planet1, planet2, planet3]
 
path = "test_data/run_figure_8/data/"
iterations = 6000
dt = 0.01
collisions = True

sim = Universe(objects, iterations, dt=dt, outpath=path, G=1)
sim.run(collisions=collisions, gpu=False)

observer_params = {"position":[0.0, 0.0, 3]}

animation = Animation(path, observer_params=observer_params)
animation.play(paused=True, track_all=True, obj_history_length=250)