from cosmosim.core.new import Object
from cosmosim.core.simulation import Simulation, Animation
import random

AU = 1.496e11   # Astronomical unit
ME = 5.972e24   # Mass of the Earth
RE = 6.371e6    # Radius of the earth
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
NUM_PLANETS = 1000
PLANET_DENSITY = 3000
D_MIN = 0.05*AU
D_MAX = 1*AU
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
iterations = 300
dt = 1e3
objects = [star, *planets]

test_sim = Simulation(objects, dt, iterations)
data = test_sim.run()

scale=2e-9
animation = Animation(data, scale=scale)
animation.play(paused=True)