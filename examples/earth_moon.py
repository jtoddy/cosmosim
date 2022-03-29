from cosmosim.core.universe import Universe 
from cosmosim.core.planet import Planet

# =============================================================================
# A model of the Earth and Moon
# =============================================================================

ME = 5.972e24       # Mass of the Earth
RE = 6.371e6        # Radius of the earth

# Create the universe
universe = Universe()

# Create the Earth
EARTH_PARAMS = {
    "mass": ME,
    "radius": RE,
    "name": "Terra",
    "position": [0,0],
    "color":  (3, 90, 252)
}
earth = Planet(**EARTH_PARAMS)
universe.add_planet(earth)

# Create the moon
MOON_PARAMS = {
    "distance": 62*RE,
    "mass": 7.3459e22,
    "radius": 1.7374e6,
    "name": "Luna",
    "color": (136, 138, 143)
}
moon = earth.create_satellite(**MOON_PARAMS)

# Create the ISS
ISS_PARAMS = {
    "distance":7e6,
    "mass": 4.5e6,
    "radius": 100,
    "name": "ISS"
}
iss = earth.create_satellite(**ISS_PARAMS)
   
#Simulate
SIMULATION_PARAMS = {
    "speed": 300,
    "scale": 1.3e-6,
    "track_all": True
}
universe.simulate(**SIMULATION_PARAMS)