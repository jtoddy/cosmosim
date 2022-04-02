from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import InteractiveAnimation
import random
import cProfile

AU = 1.496e11   # Astronomical unit
ME = 5.972e24   # Mass of the Earth
MS = 1.989e30   # Mass of the sun
DS = 1408       # Density of the sun

# Create a star
STAR_MASS = MS
STAR_DENSITY = DS
STAR_POSITION = [0,0,0]
STAR_NAME = "Sol"
STAR_COLOR = (255,255,0) # yellow

star = Object(mass=STAR_MASS, 
                density=STAR_DENSITY, 
                position=STAR_POSITION,
                name=STAR_NAME, 
                color=STAR_COLOR)

# Create some planets in a disk around the star
NUM_PLANETS = 3000
PLANET_DENSITY = 300
D_MIN = 0.1*AU
D_MAX = 0.5*AU
MIN_MASS = 0.1*ME
MAX_MASS = 100*ME

planets = []
for i in range(NUM_PLANETS):
    PLANET_DISTANCE = random.randint(D_MIN, D_MAX)
    PLANET_MASS = random.randint(MIN_MASS, MAX_MASS)
    p = star.create_satellite(distance=PLANET_DISTANCE,
                              mass=PLANET_MASS,
                              density=PLANET_DENSITY)
    planets.append(p)
   
#Simulate
path = "test_data/profiler_run/data/"
iterations = 100
dt = 600
objects = [star, *planets]
collisions = True
scale=6.5e-9

test_sim = Universe(objects, dt, iterations, path+"cpu/")
cProfile.run("test_sim.run(collisions=collisions, gpu=False)", "restats_gen")
animation = InteractiveAnimation(path+"cpu/", scale=scale)
animation.play()

test_sim_gpu = Universe(objects, dt, iterations, path+"gpu/")
cProfile.run("test_sim_gpu.run(collisions=collisions, gpu=True)", "restats_gen_gpu")
animation_gpu = InteractiveAnimation(path+"gpu/", scale=scale)
animation_gpu.play()