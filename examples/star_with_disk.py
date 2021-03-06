from cosmosim.core.universe import Universe
from cosmosim.util.functions import get_radius
import random

# Create the universe
universe = Universe()

# Create a star
STAR_DENSITY = 1
STAR_MASS = 300000
STAR_RADIUS = get_radius(STAR_MASS, STAR_DENSITY)
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
PLANET_DENSITY = 1
D_MAX = 1000
D_MIN = 300
MIN_MASS = 1
MAX_MASS = 10

for i in range(NUM_PLANETS):
    PLANET_DISTANCE = random.randint(D_MIN, D_MAX)
    PLANET_MASS = random.randint(MIN_MASS, MAX_MASS)
    PLANET_RADIUS = get_radius(PLANET_MASS, PLANET_DENSITY)
    star.create_satellite(distance=PLANET_DISTANCE,
                          mass=PLANET_MASS,
                          radius=PLANET_RADIUS)
   
#Simulate
universe.simulate()