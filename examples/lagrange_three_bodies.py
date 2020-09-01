from cosmosim.core.universe import Universe 
from cosmosim.core.planet import Planet
import cosmosim.util.functions as F
import math
import numpy as np

# =============================================================================
# Lagrange's solution to the 3-body problem.
# Not perfect in this simulator, but it stays stable for a while.
# Could probably be tuned a bit more
# =============================================================================

# Create the universe
G = 1
universe = Universe(G=G)

#Add planets
pi = math.pi
angles = [0.0, 2*pi/3, 4*pi/3]
r = 100
m = 10000
omega = math.sqrt((G*m)/(math.sqrt(3)*r**3))
print("Angular velocity: %f" % omega)
for theta in angles:
    p = F.to_cartesian(r, theta)
    v = F.rotation(omega*np.array(p), pi/2)
    planet = Planet(mass=m, radius=10, position=p, velocity=v)
    universe.add_planet(planet)

#Simulate
universe.simulate(track_all=True)