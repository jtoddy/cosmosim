from cosmosim.core.universe import Universe 
from cosmosim.core.planet import Planet
import cosmosim.util.functions as F

# =============================================================================
# A "roughly to scale" model of the solar system.
# Moons and pluto are not included.
# =============================================================================
 
# Create the universe
universe = Universe()

# Create the sun
STAR_DENSITY = 1/4
STAR_MASS = 333000
STAR_RADIUS = F.get_radius(STAR_MASS, STAR_DENSITY)
STAR_POSITION = [0,0]
STAR_NAME = "Sol" 
STAR_COLOR = (255,255,0)
IMMOBILE = True
sol = Planet(STAR_MASS, STAR_RADIUS, STAR_POSITION, immobile=IMMOBILE, name=STAR_NAME, color=STAR_COLOR)
universe.add_planet(sol)

au = STAR_RADIUS*215

# Create the planets
planets = [
    #Mercury
    {"name": "Mercury",
     "distance": 0.4*au,
     "mass": 1,
     "radius": 1,
     "color":(173, 152, 125)},
    
    #Venus
    {"name": "Venus",
     "distance": 0.7*au,
     "mass": 1,
     "radius": 1,
     "color":(255, 240, 153)},
    
    #Earth
    {"name": "Earth",
     "distance": 1*au,
     "mass": 1,
     "radius": 1,
     "color":(52, 55, 235)},
    
    #Mars
    {"name": "Mars",
     "distance": 1.5*au,
     "mass": 1,
     "radius": 1,
     "color":(255, 51, 0)},
    
    #Jupiter
    {"name": "Jupiter",
     "distance": 5.2*au,
     "mass": 320,
     "radius": 11,
     "color":(255, 51, 0)},
    
    #Saturn
    {"name": "Saturn",
     "distance": 9.5*au,
     "mass": 95,
     "radius": 9,
     "color":(255, 227, 117)},
    
    #Uranus
    {"name": "Uranus",
     "distance": 19*au,
     "mass": 15,
     "radius": 4,
     "color":(130, 209, 209)},
    
    #Neptune
    {"name": "Neptune",
     "distance": 30*au,
     "mass": 17,
     "radius": 4,
     "color":(36, 24, 201)}
    ]

for planet in planets:
    sol.create_satellite(**planet)

#Simulate 
SIMULATION_PARAMS = {
    "speed": 300,
    "scale": 2e-6,
    "track_all": True
}
universe.simulate()

