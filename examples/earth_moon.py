from cosmosim.core.universe import Universe 
from cosmosim.core.planet import Planet
import cosmosim.util.functions as F

# Create the universe
universe = Universe()

# Create the Earth
EARTH_DENSITY = 1
EARTH_MASS = 8000
EARTH_RADIUS = F.get_radius(EARTH_MASS, EARTH_DENSITY)
EARTH_POSITION = [0,0]
EARTH_NAME = "Terra"
EARTH_COLOR = (3, 90, 252)
IMMOBILE = True
EARTH = Planet(universe, EARTH_MASS, EARTH_RADIUS, EARTH_POSITION, immobile=IMMOBILE, name=EARTH_NAME, color=EARTH_COLOR)

# Create the moon
MOON_DISTANCE = 62*EARTH_RADIUS
MOON_MASS = EARTH_MASS/80
MOON_RADIUS = F.get_radius(MOON_MASS, 1)
MOON_NAME = "Luna"
MOON_COLOR = (136, 138, 143)
EARTH.create_satellite(distance=MOON_DISTANCE,mass=MOON_MASS,radius=MOON_RADIUS,name=MOON_NAME, color=MOON_COLOR)
   
#Simulate
universe.simulate()