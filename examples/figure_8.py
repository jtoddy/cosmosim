from cosmosim.core.universe import Universe 

# =============================================================================
# Figure-8 solution to the 3-body problem
# =============================================================================

# Create the universe
universe = Universe()

# Create planets
ME = 5.972e18      # Mass of the Earth
RE = 6.371       # Radius of the earth
k = 100

p1 = [0.97000436*k*RE, -0.24308753*k*RE] 
p2 = [-p1[0], -p1[1]] 
p3 = [0, 0]

v3 = [-0.93240737,-0.86473146] 
v2 = [-v3[0]/2,-v3[1]/2]
v1 = v2

planet1 = universe.create_planet(mass=ME, radius=RE, position=p1, velocity=v1)
planet2 = universe.create_planet(mass=ME, radius=RE, position=p2, velocity=v2)
planet3 = universe.create_planet(mass=ME, radius=RE, position=p3, velocity=v3)
 
#Simulate
universe.simulate(speed=100, track_all=True,scale=1, start_paused=True)