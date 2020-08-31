# cosmosim
 
An n-body simulator implemented in Pygame. 

## Basic functionality

Create the universe:

```
from cosmosim.core.universe import Universe 
from cosmosim.core.planet import Planet

universe = Universe()
```

Create a planet and add it to the universe:

```
planet1 = Planet(mass=1000, radius=6, position=[0,0])
universe.add_planet(planet1)
```

Or, add a planet directly to the universe upon creation:
```
planet2 = universe.create_planet(mass=10, radius=2, position=[100,0], velocity=[0,2])
```

Create a planet with randomized parameters:

```
planet3 = universe.random_planet()
```

Add a satellite around the random planet:

```
planet3a = planet3.create_satellite()
```

Run the simulation:

```
universe.simulate()

```
## Example

Cosmosim model interactions between hundreds of objects:

```
from cosmosim.core.universe import Universe
import cosmosim.util.functions as F
import math
 
# Create the universe
universe = Universe()

# Create some planets
NUM_PLANETS = 1000
omega = 0.03 # angular velocity

for i in range(NUM_PLANETS):
    # Create random planet
    planet = universe.random_planet()
    # Spin it around the origin a bit
    planet.velocity = F.rotation(planet.position*omega, math.pi/2)
   
#Simulate
universe.simulate()
```

## Controls

You can move the screen by clicking and dragging with the left mouse button, or by using the `WASD` keys.

The mouse wheel controls zoom.

`SPACEBAR` pauses the simulation.

`+` and `-` keys change the speed.

Right click resets zoom, speed, and screen position.

Clicking on a planet will bring up some of its stats and give you the option to track or destroy it.
