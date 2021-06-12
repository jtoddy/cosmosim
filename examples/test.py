from cosmosim.core.new import Object
from cosmosim.core.simulation import Simulation, Animation
import random
import math
import cosmosim.util.functions as F

WHITE = (255,255,255)
YELLOW = (255,255,0)
BLACK = (0,0,0)

AU = 1.496e11       # Astronomical unit
ME = 5.972e24       # Mass of the Earth
RE = 6.371e6        # Radius of the earth
MS = 1.989e30       # Mass of the sun
RS = 6.9634e8       # Radius of the sun
DAYTIME = 86400     # Seconds in a day
_G = 6.674e-11      # Gravitational constant

def random_object(density=4000, min_mass=0.1*ME, max_mass=1000*ME, dmin=0, 
                  dmax=50*AU, min_velocity=0, max_velocity=5e4, 
                  name=None, color=None):
    mass = random.randint(min_mass, max_mass)
    pspherical = [
        random.randint(dmin,dmax), 
        random.random()*(2*math.pi), 
        random.random()*(math.pi)
    ]
    position = F.to_cartesian_3d(*pspherical)
    v0 = min_velocity + (max_velocity - min_velocity) * random.random()
    v_theta = 2*math.pi * random.random()
    v_phi = math.pi * random.random()
    v_polar = [v0, v_theta, v_phi]
    velocity = F.to_cartesian_3d(*v_polar)
    obj = Object(
        mass=mass, 
        density=density, 
        position=position, 
        velocity=velocity,
        color=color,
        name=name
    )
    return obj

iterations = 2000
dt = 1e6
scale = 5e-11
objects = []
path = "C:/test_data/cosmosim/test_run/"

# Create some planets
# NUM_OBJECTS = 1000
# omega = 1e-9 # angular velocity
# for i in range(NUM_OBJECTS):
#     # Create random object
#     obj = random_object()
#     # Spin it around the origin a bit
#     obj.velocity = F.rotation_3d(obj.position*omega, math.pi/2, 0)
#     objects.append(obj)
   
# print("Generating data...")
# test_sim = Simulation(objects, dt, iterations, path)
# test_sim.run()
# print("Done!")

animation = Animation(path, scale=scale)
animation.play()