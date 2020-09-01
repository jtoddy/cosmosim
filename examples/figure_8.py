from cosmosim.core.universe import Universe 

# =============================================================================
# Figure-8 solution to the 3-body problem
# =============================================================================

# Create the universe
universe = Universe()

# Create planets
m = 100
r = 7
k = 100

p1 = [0.97000436*k, -0.24308753*k] 
p2 = [-p1[0], -p1[1]] 
p3 = [0, 0]

v3 = [-0.93240737,-0.86473146] 
v2 = [-v3[0]/2,-v3[1]/2]
v1 = v2

planet1 = universe.create_planet(mass=m, radius=r, position=p1, velocity=v1)
planet2 = universe.create_planet(mass=m, radius=r, position=p2, velocity=v2)
planet3 = universe.create_planet(mass=m, radius=r, position=p3, velocity=v3)

#Simulate
universe.simulate(speed=20, track_all=True)