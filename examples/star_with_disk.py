from cosmosim.core.universe import Universe
from cosmosim.util.functions import get_radius
import random

AU = 1.496e11   # Astronomical unit
ME = 5.972e24   # Mass of the Earth
RE = 6.371e6    # Radius of the earth
MS = 1.989e30   # Mass of the sun
RS = 6.9634e8   # Radius of the sun

# Create the universe
universe = Universe()

# Create a star
STAR_MASS = MS
STAR_RADIUS = RS
STAR_POSITION = [0,0]
STAR_NAME = "Sol"
STAR_COLOR = (255,255,0) # yellow
IMMOBILE = True

star = universe.create_planet(mass=STAR_MASS, 
                              radius=STAR_RADIUS, 
                              position=STAR_POSITION, 
                              immobile=IMMOBILE,
                              name=STAR_NAME, 
                              color=STAR_COLOR)

# Create some planets in a disk around the star
NUM_PLANETS = 1000
PLANET_DENSITY = 3000
D_MIN = 0.05*AU
D_MAX = 1*AU
MIN_MASS = 0.1*ME
MAX_MASS = 100*ME

for i in range(NUM_PLANETS):
    PLANET_DISTANCE = random.randint(D_MIN, D_MAX)
    PLANET_MASS = random.randint(MIN_MASS, MAX_MASS)
    PLANET_RADIUS = get_radius(PLANET_MASS, PLANET_DENSITY)
    star.create_satellite(distance=PLANET_DISTANCE,
                          mass=PLANET_MASS,
                          radius=PLANET_RADIUS)
   
#Simulate
universe.simulate(scale=2e-9)