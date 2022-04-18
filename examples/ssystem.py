import sys
import os
sys.path.insert(0, os.getcwd())

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

# Solar system state vectors on 2021-01-01 00:00:00
planet_stats = {
	#Planets
	"mercury":{
		"mass":0.33e24,
		"density":5429,
		"color":[186,125,71],
		"position":[3.446368089632962e07, -5.265373901017199e07, -7.612893209049892e06],
		"velocity":[3.088457353159489e01,  2.926845255623352e01, -4.412682585863976e-01]
	},
	"venus":{
		"mass":4.87e24,
		"density":5243,
		"color":[189,189,22],
		"position":[-6.777610420819579e07, -8.436124558209643e07,  2.699450587664146e06],
		"velocity":[2.731307492362905e01, -2.176908890925689e01, -1.875042568207816e00]
	},
	"earth":{
		"mass":5.97e24,
		"density":5514,
		"color":[22,106,189],
		"position":[-2.779255047284979e07,  1.455308961614190e08,  8.814210802167654e03],
		"velocity":[-2.977734693025362e01, -5.542405659101232e00, -8.551711677950991e-05]
	},
	"mars":{
		"mass":0.642e24,
		"density":3934,
		"color":[189,61,22],
		"position":[9.188516192255516e07,  2.066968932224656e08,  2.050077368338853e06],
		"velocity":[-2.117849969498003e01,  1.201689386467743e01,  7.716520426787934e-01]
	},
	"jupiter":{
		"mass":1898e24,
		"density":1326,
		"color":[255,86,114],
		"position":[4.540045001070552e08, -6.106818485721847e08, -7.623924721314132e06],
		"velocity":[1.032238837192197e01,  8.414069532136335e00, -2.659049139509011e-01]
	},
	"saturn":{
		"mass":568e24,
		"density":687,
		"color":[225,190,52],
		"position":[8.203515903205494e08, -1.247065425479232e09, -1.097619517676431e07],
		"velocity":[7.531925768776626e00,  5.285843754677155e00, -3.924010739117183e-01]
	},
	"uranus":{
		"mass":86.8e24,
		"density":1270,
		"color":[52,179,225],
		"position":[2.295337152343883e09,  1.865127568978773e09, -2.280932961710787e07],
		"velocity":[-4.344309294719951e00,  4.967783963505782e00,  7.467913435812190e-02]
	},
	"nepturne":{
		"mass":102e24,
		"density":1638,
		"color":[52,58,225],
		"position":[4.406236886270197e09, -7.817713304515282e08, -8.544717467575288e07],
		"velocity":[9.128317569628802e-01,  5.384083139759752e00, -1.312921917550081e-01]
	},
	# Moons
	"luna":{
		"mass":0.073e24,
		"density":3340,
		"color":[137,137,137],
		"position":[-2.799949483881730e07,  1.458564063289887e08,  3.288263030440360e04],
		"velocity":[-3.061392327004896e01, -6.124622492111070e00,  6.582757972820241e-02]
	},
	"io":{
		"mass":8.931e22,
		"density":3528,
		"color":[247,206,41],
		"position":[4.540829581261586E+08, -6.110974757657857E+08, -7.637855216257393E+06],
		"velocity":[2.730642503970988E+01,  1.156728504001686E+01,  1.063129436279153E-01]
	},
	"europa":{
		"mass":4.799e22,
		"density":3013,
		"color":[168,197,216],
		"position":[4.534029247735934E+08, -6.109829880145832E+08, -7.648929156198651E+06],
		"velocity":[1.657494406123974E+01, -3.769574338777018E+00, -6.047082463597186E-01]
	},
	"ganymede":{
		"mass":1.4819e23,
		"density":1936,
		"color":[120,128,134],
		"position":[4.538394253345867E+08, -6.117381985877813E+08, -7.666353234886587E+06],
		"velocity":[2.107935484038133E+01,  6.745987008155156E+00, -1.823217594996787E-01]
	},
	"callisto":{
		"mass":1.075e23,
		"density":1834,
		"color":[115,69,45],
		"position":[4.549264289025569E+08, -6.123168960727901E+08, -7.663156457886398E+06],
		"velocity":[1.746380116455424E+01,  1.249635020663894E+01, -4.082729452819400E-02]
	},
	"titan":{
		"mass":1.345e23,
		"density":1880,
		"color":[230,186,10],
		"position":[8.192118450254436E+08, -1.246557931437262E+09, -1.112429427030587E+07],
		"velocity":[5.278239460935746E+00,  1.002170842064872E+00,  2.040371389848600E+00]
	}
}

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

# test_sim = Universe([sun]+planets, iterations, dt=dt, outpath=path)
# test_sim.run(collisions=collisions, gpu=False)

animation = Animation(path, observer_params=observer_params)
animation.play()