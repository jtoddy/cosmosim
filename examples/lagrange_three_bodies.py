from cosmosim.core.universe import Universe 
from cosmosim.core.planet import Planet
import cosmosim.util.functions as F
import math
import numpy as np

# Create the universe
G = 0.1
universe = Universe(G=G)


#Add planets
pi = math.pi
angles = [0.0, 2*pi/3, 4*pi/3]
r = 100
omega = 0.025
for theta in angles:
    p = F.to_cartesian(r, theta)
    v = F.rotation(omega*np.array(p), pi/2)
    planet = Planet(universe, mass=10000, radius=5, position=p, velocity=v)
    universe.add_planet(planet)

#Simulate
universe.simulate()