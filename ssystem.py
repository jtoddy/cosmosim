from cosmosim.core.universe import Object, Universe
from cosmosim.core.animation import Animation
from cosmosim.util.constants import AU, ME, DE, MS, DS
import cosmosim.util.functions as F
import random
import math
import numpy as np
import json

# Create a star
STAR_MASS = MS
STAR_DENSITY = DS
STAR_POSITION = [0,0,0]
STAR_NAME = "Sol"
STAR_COLOR = (255,255,0) # yellow

sun = Object(
	mass=STAR_MASS, 
    density=STAR_DENSITY, 
    position=STAR_POSITION,
    name=STAR_NAME, 
    color=STAR_COLOR
)

f = open("ssystem.json")
planet_stats = json.load(f)

planets = []
for planet_name in planet_stats:
	stats = planet_stats[planet_name]
	mass = stats["mass"]
	density = stats["density"]
	color = stats["color"]
	position = np.array(stats["position"])*1000
	velocity = np.array(stats["velocity"])*1000
	obj = Object(
		name=planet_name,
		mass=mass,
		density=density,
		color=color,
		position=position,
		velocity=velocity
	)
	planets.append(obj)

path = "test_data/run_ssystem/data/"
iterations = 6000
dt = 60
collisions = True
observer_position = [0.0, 0.0, 5*AU]
observer_params = {"position":observer_position, "theta":0.0, "phi":0.0}

test_sim = Universe([sun]+planets, iterations, dt=dt, outpath=path, filesize=3000)
test_sim.run(collisions=collisions, gpu=False)

animation = Animation(path, observer_params=observer_params)
animation.play()