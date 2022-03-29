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

ME = 5.972e24       # Mass of the Earth
RE = 6.371e6        # Radius of the earth

# Create the universe
universe = Universe()
G = universe.G

#Add planets
pi = math.pi
angles = [0.0, 2*pi/3, 4*pi/3]
r = 10*RE
omega = math.sqrt((G*ME)/(math.sqrt(3)*r**3))
for theta in angles:
    p = F.to_cartesian(r, theta)
    v = F.rotation(omega*np.array(p), pi/2)
    planet = Planet(mass=ME, radius=RE, position=p, velocity=v)
    universe.add_planet(planet)
    
#Simulate
SIMULATION_PARAMS = {
    "speed": 1e4,
    "scale": 3.51e-6,
    "track_all": True
}
universe.simulate(**SIMULATION_PARAMS)