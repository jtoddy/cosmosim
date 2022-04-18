import sys
import os
sys.path.insert(0, os.getcwd())

from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import Animation
from cosmosim.util.constants import AU, ME, DE, MS, DS
import cosmosim.util.functions as F
import random
import math
import numpy as np

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
NUM_PLANETS = 1000
PLANET_DENSITY = 3000
D_MIN = 0.1*AU
D_MAX = 0.5*AU
MIN_MASS = 0.1*ME
MAX_MASS = 100*ME

planets = []
for i in range(NUM_PLANETS):
    PLANET_DISTANCE = random.randint(D_MIN, D_MAX)
    PLANET_MASS = random.randint(MIN_MASS, MAX_MASS)
    keplerian_elements = {
    	# Semi-major axis
    	"a":PLANET_DISTANCE,
    	# Inclination
    	"i": np.random.uniform(0, 1/16)*math.pi,
    	# Argument of apoapsis
    	"omega_AP": np.random.uniform(0, 2)*math.pi,
    	# Longitude of ascending node
    	"omega_LAN": np.random.uniform(0, 2)*math.pi,
    }
    p = star.create_satellite(keplerian_elements=keplerian_elements,
                              mass=PLANET_MASS,
                              density=PLANET_DENSITY)
    planets.append(p)
   
path = "test_data/run_kep/data/"
iterations = 6000
dt = 600
collisions = True
observer_position = [0.0, 0.0, AU]
observer_params = {"position":observer_position, "theta":0.0, "phi":0.0}

# test_sim = Universe([star]+planets, iterations, dt=dt, outpath=path, filesize=3000)
# test_sim.run(collisions=collisions, gpu=False)

animation = Animation(path, observer_params=observer_params)
animation.play()