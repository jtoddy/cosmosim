import pygame
import numpy as np
import random
import math
import cosmosim.util.functions as F
import cosmosim.util.pronounceable.main as prnc

WHITE = (255,255,255)
YELLOW = (255,255,0)
BLACK = (0,0,0)

class Planet:
    
    def __init__(self, mass, radius, position, velocity=[0.0,0.0], color=None, immobile=False, name=None, universe=None):
        # Initialize planet
        self.universe = None
        self.name = name
        self.mass = mass
        self.radius = radius
        self.volume = ((4/3)*math.pi)*(self.radius**3)
        self.density = self.mass/self.volume
        self.position = np.array(position).astype(float)
        self.velocity = np.array(velocity).astype(float)
        self.energy = 0
        self.color = color
        self.immobile = immobile
        self.history = []
        self.bound = False
        self.clicked = False
        self.tracked = False
        self.alive = True
        self.planets_eaten = 0
        if not self.name:
            self.name = prnc.generate_word()
        if not self.color:
            self.color = (int(255*random.random()),int(255*random.random()),int(255*random.random()))
            
    def create_satellite(self, distance=None, mass=None, radius=None, theta=None, name=None, color=None):
        G = self.universe.G
        if not distance:
            distance = random.randint(int(self.radius*5), int(self.radius*100))
        if not mass:
            mass = random.random()*self.mass
        if not radius:
            radius = F.get_radius(mass, 1)
        if theta is None:
            theta = 2*math.pi*random.random()
        v_mag = math.sqrt((self.mass*G)/distance) # Circular orbit
        pos = F.to_cartesian(distance, theta)[::-1]
        pos_norm = pos/np.linalg.norm(pos)
        v = F.rotation(v_mag*pos_norm,math.pi/2)
        return self.universe.create_planet(mass=mass,
                                           radius=radius,
                                           position=pos,
                                           velocity=v,
                                           name=name,
                                           color=color)
            
    def update_history(self, trail_length):
        if self.alive and self.tracked:
            pos = np.array([self.position[0],self.position[1]])
            self.history.append(pos)
        else:
            self.history = []
        if len(self.history) > trail_length:
            self.history.remove(self.history[0])
                
    def absorb(self, planet):
        if self.alive and planet.alive:
            if not self.immobile:
                # Inelastic collision
                self.position = ((self.mass*self.position)+(planet.mass*planet.position))/(self.mass+planet.mass)
                self.velocity = ((self.mass*self.velocity)+(planet.mass*planet.velocity))/(self.mass+planet.mass)
            self.mass += planet.mass
            self.volume += planet.volume
            self.radius = ((3*self.volume)/(4*math.pi))**(1/3)
            planet.destroy()
            self.planets_eaten += 1 + planet.planets_eaten
        
    def destroy(self):
        self.alive = False
        self.velocity = np.array([0,0])
        self.acceleration = np.array([0,0])
        self.position = None
        self.energy = 0
     
    def draw(self, screen, color):
        scale = self.universe.context['scale']
        # Set the radius of the planet
        radius = int(self.radius*scale)
        # Draw
        if self.alive:
            q = F.screen_coordinates(self.position, **self.universe.context) 
            if self.clicked:
                # Highlight the planet
                 pygame.draw.circle(screen, WHITE, q.astype(int), max(int(radius*2), 4)) 
                 pygame.draw.circle(screen, BLACK, q.astype(int), max(int(radius*1.85), 2))
            if self.tracked:
                 # Trace the planet's path
                 h_length = len(self.history)
                 #for i, p in enumerate(self.history):
                 for i in range(h_length):
                     if i > 0:
                         p0 = self.history[i-1]
                         p = self.history[i]
                         alpha = i/h_length
                         trail_color = tuple(alpha*i for i in self.color)
                         q = F.screen_coordinates(p, **self.universe.context)
                         q0 = F.screen_coordinates(p0, **self.universe.context)
                         pygame.draw.line(screen, trail_color, q0.astype(int), q.astype(int), 1)
            # Draw the planet itself
            pygame.draw.circle(screen, color, q.astype(int), radius)   
    