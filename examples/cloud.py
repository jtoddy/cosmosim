from cosmosim.core.universe import Universe
import cosmosim.util.functions as F
import math
 
# Create the universe
universe = Universe()

# Create some planets
NUM_PLANETS = 1000
omega = 0.02 # angular velocity

for i in range(NUM_PLANETS):
    # Create random planet
    planet = universe.random_planet()
    # Spin it around the origin a bit
    planet.velocity = F.rotation(planet.position*omega, math.pi/2)
   
#Simulate
universe.simulate()