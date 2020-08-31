import pygame
import numpy as np
import random
import math
import cosmosim.util.functions as F
from cosmosim.util.blas import acc_blas
from cosmosim.core.planet import Planet
from sklearn.metrics import pairwise_distances

WHITE = (255,255,255)
YELLOW = (255,255,0)
BLACK = (0,0,0)

class Universe:
    
    def __init__(self, G=1):
        self.G = G
        self.planets = []
        self.angular_momenta = np.array([0])
        self.clicked_planet = None
        self.running = False
        self.paused = False
        self.dragging = False
        # initialize buttons
        self.close_btn = None
        self.track_btn = None
        self.destroy_btn = None
                
    def add_planet(self, planet):
        self.planets.append(planet)
        planet.universe = self
        
    def create_planet(self, **kwargs):
        planet = Planet(**kwargs)
        self.add_planet(planet)
        return planet
        
    def random_planet(self, density=1, min_mass=1, max_mass=100, dmin=0, dmax=500, min_velocity=1.0, max_velocity=3.0, name=None, color=None):
        mass = random.randint(min_mass, max_mass)
        radius = F.get_radius(mass,density)
        ppolar = [random.randint(dmin,dmax), random.random()*(2*math.pi)]
        position = F.to_cartesian(*ppolar)
        v0 = min_velocity + (max_velocity - min_velocity) * random.random()
        v_theta = 2*math.pi * random.random()
        velocity = [v0*math.cos(v_theta),v0*math.sin(v_theta)]
        return self.create_planet(mass=mass,radius=radius,position=position,velocity=velocity,color=color,name=name)
    
    def active_planets(self):
        return [planet for planet in self.planets if planet.alive]
    
    def absorbed_planets(self):
        return [planet for planet in self.planets if not planet.alive]
    
    def bound_planets(self):
        return [planet for planet in self.active_planets() if planet.energy < 0]

    def unbound_planets(self):
        return [planet for planet in self.active_planets() if planet.energy >= 0]
    
    def net_angular_momentum(self):
        return abs(np.sum(self.angular_momenta))
    
    def simulate(self, width=1600, height=1000, speed=1, fps=33, trail_length=1000, collisions=True):
        
        self.context = {"scale":1.0, 
                        "offset":np.array([0.0,0.0]), 
                        "origin":np.array([width/2,height/2])}
        
        # Set up display
        pygame.init()
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
        screen = pygame.display.set_mode([width, height])
        pygame.display.set_caption('cosmosim 0.10')
        
        # Run simulation
        dt = speed/20
        dt_default = dt
        self.running = True
        while self.running:
            
            # Clear the screen
            screen.fill(BLACK)
            
            # Handle user inputs
            for event in pygame.event.get():
                # Stop simulation when user quits
                if event.type == pygame.QUIT:
                    self.running = False
                # Key presses
                elif event.type == pygame.KEYDOWN:
                    # Spacebar pauses
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    # WASD moves the screen
                    if event.key == pygame.K_w:
                        self.context['offset'] += np.array([0,50])/self.context['scale']
                    if event.key == pygame.K_a:
                        self.context['offset'] += np.array([50,0])/self.context['scale']
                    if event.key == pygame.K_s:
                        self.context['offset'] += np.array([0,-50])/self.context['scale']
                    if event.key == pygame.K_d:
                        self.context['offset'] += np.array([-50,00])/self.context['scale']
                    # +/- keys change simulation speed
                    if event.key == pygame.K_KP_PLUS:
                        dt *= 1.05
                    if event.key == pygame.K_KP_MINUS:
                        dt *= 0.95
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Left click drags view
                    if event.button == 1:
                        self.dragging = True
                        mouse_x0, mouse_y0 = event.pos 
                    # Right click resets view
                    elif event.button == 3:
                        self.context['offset']  = np.array([0.0,0.0])
                        self.context['scale']  = 1.0
                        dt = dt_default
                    # Wheel up zooms in 5%
                    elif event.button == 4:
                        self.context['scale']  *= 1.05 
                    # Wheel down zooms out 5%
                    elif event.button == 5:
                        self.context['scale']  *= 0.95
                    else:
                        pass
                elif event.type == pygame.MOUSEBUTTONUP:
                    # Releasing left click
                    if event.button == 1:            
                        self.dragging = False
                        pos = pygame.mouse.get_pos()
                        # Click on close button
                        if self.close_btn and self.close_btn.collidepoint(pos):
                            self.clicked_planet.clicked = False
                            self.clicked_planet = None
                        # Click on tracking button
                        elif self.track_btn and self.track_btn.collidepoint(pos):
                            self.clicked_planet.tracked = not self.clicked_planet.tracked
                        # Click on destroy button
                        elif self.destroy_btn and self.destroy_btn.collidepoint(pos):
                            self.clicked_planet.clicked = False
                            self.clicked_planet.tracked = False
                            self.clicked_planet.destroy()
                            self.clicked_planet = None
                        # Click on planet
                        else:
                            clicked_planets = [p for p in self.active_planets() if np.linalg.norm(F.screen_coordinates(p.position, **self.context) - pos) <= p.radius*self.context['scale']]
                            if len(clicked_planets) > 0:
                                if self.clicked_planet:
                                    self.clicked_planet.clicked = False
                                self.clicked_planet = clicked_planets[0]
                                self.clicked_planet.clicked = True
                elif event.type == pygame.MOUSEMOTION:
                    # Dragging mouse moves the screen
                    if self.dragging:
                        mouse_x, mouse_y = event.pos
                        dx = int((mouse_x - mouse_x0)/(5*self.context['scale']))
                        dy = int((mouse_y - mouse_y0)/(5*self.context['scale']))
                        self.context['offset'] += np.array([dx,dy])
                else:
                    pass
            
            # Simulate gravitational interactions between planets
            active_planets = self.active_planets()
            if not self.paused:
                # Initial conditions
                m = np.array([p.mass for p in active_planets])
                p0 = np.array([p.position for p in active_planets])
                v0 = np.array([p.velocity for p in active_planets])
                # Calculate net accelerations
                a_3d = acc_blas(p0, m, self.G)  # Magic!!!
                a = np.delete(a_3d, 2, 1)       # Only need 2D acceleration
                # Integration
                v = v0 + a*dt
                p = p0 + v*dt
                # Calculate energies and gravitational potential
                a_mag = np.linalg.norm(a, axis=1)
                V = -np.sqrt(self.G*a_mag*(np.sum(m)-m))
                U = m * V
                K = 0.5*m*(np.linalg.norm(v, axis=1)**2)
                # Calculate angular momenta
                self.angular_momenta = p*(np.dot(v, [F.get_tangent(p) for p in p]))*m[:,np.newaxis]
                # Identify collisions
                if collisions:
                    r = np.array([p.radius for p in active_planets])
                    d = pairwise_distances(p)
                    collision_matrix = d <= np.add.outer(r,r)
                
            # Update planet properties and draw
            for i, planet in enumerate(active_planets):
                if not self.paused:
                    if planet.alive:
                        if not planet.immobile:
                            planet.velocity = v[i]
                            planet.position = p[i]
                        if collisions:
                            for j, c in enumerate(collision_matrix[i]):
                                if c and i != j:
                                    other_planet = active_planets[j]
                                    if planet.mass >= other_planet.mass:
                                        planet.absorb(other_planet)
                                    else:
                                        other_planet.absorb(planet)
                        planet.energy =  K[i] + U[i]
                    planet.update_history(trail_length)
                planet.draw(screen, planet.color)
                
                
            # Text and menus

            # Update universe info text
            universe_text = [
            "Total Planets: " + str(len(self.active_planets())), 
            "Bound Planets: " + str(len(self.bound_planets())),
            "Escaped Planets: " + str(len(self.unbound_planets())),
            "Destroyed Planets: " + str(len(self.absorbed_planets())),
            "Net Angular Momentum: {:.2e}".format(self.net_angular_momentum()),
            ]
            
            for i, text in enumerate(universe_text):
                img = font.render(text, True, WHITE)
                screen.blit(img, (20, (20)*(i+1)))
            
            # Update clicked planet text (MAKE PLANET METHOD?)
            if self.clicked_planet:
                if self.clicked_planet.alive:
                    # Info box dimensions
                    INFO_X = 10
                    INFO_Y = 130
                    INFO_WIDTH = 240
                    INFO_HEIGHT = 150
                    BTN_SIZE = 30
                    TXT_PAD = 10
                    INFO_X2 = INFO_X + INFO_WIDTH
                    INFO_Y2 = INFO_Y + INFO_HEIGHT
                    
                    # Box Outline
                    pygame.draw.rect(screen, WHITE, ((INFO_X, INFO_Y), (INFO_WIDTH, INFO_HEIGHT)), 2)
                    
                    # Header bar
                    pygame.draw.line(screen, WHITE, (INFO_X,INFO_Y+BTN_SIZE), (INFO_X2,INFO_Y+BTN_SIZE), 1)
                    
                    # Close button
                    self.close_btn = pygame.draw.rect(screen, WHITE, ((INFO_X2-BTN_SIZE,INFO_Y),(BTN_SIZE,BTN_SIZE)), 2)
                    close_img = font.render("X", True, WHITE)
                    screen.blit(close_img, (INFO_X2-BTN_SIZE+TXT_PAD,INFO_Y+TXT_PAD))
                    
                    # Track button
                    if self.clicked_planet.tracked:                  
                        self.track_btn = pygame.draw.rect(screen, WHITE, ((INFO_X,INFO_Y2),(INFO_WIDTH/2,BTN_SIZE)), 0)
                        track_img = font.render("TRACK", True, BLACK)
                    else:
                        self.track_btn = pygame.draw.rect(screen, WHITE, ((INFO_X,INFO_Y2),(INFO_WIDTH/2,BTN_SIZE)), 2)
                        track_img = font.render("TRACK", True, WHITE)
                    screen.blit(track_img, (INFO_X+TXT_PAD, INFO_Y2+TXT_PAD-1))
                    
                    # Destroy button
                    self.destroy_btn = pygame.draw.rect(screen, WHITE, ((INFO_X+(INFO_WIDTH/2),INFO_Y2),(INFO_WIDTH/2,BTN_SIZE)), 2)
                    destroy_img = font.render("DESTROY", True, WHITE)
                    screen.blit(destroy_img, (INFO_X+(INFO_WIDTH/2)+TXT_PAD, INFO_Y2+TXT_PAD-1))
                    
                    # Planet info text
                    ptext1 = self.clicked_planet.name.upper()
                    ptext2 = "Mass: %.0f" % round(self.clicked_planet.mass,2) 
                    ptext3 = "Radius: %.2f" % self.clicked_planet.radius
                    ptext4 = "Velocity: [%.2f %.2f] %.2f" % (*self.clicked_planet.velocity, np.linalg.norm(self.clicked_planet.velocity))
                    ptext5 = "Energy: {:.2e}".format(round(self.clicked_planet.energy, 2))
                    ptext6 = "Planets eaten: %i" % self.clicked_planet.planets_eaten
                    pimg1 = font.render(ptext1, True, WHITE)
                    pimg2 = font.render(ptext2, True, WHITE)
                    pimg3 = font.render(ptext3, True, WHITE) 
                    pimg4 = font.render(ptext4, True, WHITE)
                    pimg5 = font.render(ptext5, True, WHITE)
                    pimg6 = font.render(ptext6, True, WHITE)
                    screen.blit(pimg1, (20, INFO_Y + TXT_PAD))
                    screen.blit(pimg2, (20, INFO_Y + 2*TXT_PAD + 20))
                    screen.blit(pimg3, (20, INFO_Y + 2*TXT_PAD + 40))
                    screen.blit(pimg4, (20, INFO_Y + 2*TXT_PAD + 60))
                    screen.blit(pimg5, (20, INFO_Y + 2*TXT_PAD + 80))
                    screen.blit(pimg6, (20, INFO_Y + 2*TXT_PAD + 100))
                else:
                    self.clicked_planet.clicked = False
                    self.clicked_planet = None
            
            # Update FPS
            P = (clock.get_time() + 1)/1000
            effective_fps = 1/P
            fps_text = "FPS: %.0f" % effective_fps
            fps_img = font.render(fps_text, True, WHITE)
            screen.blit(fps_img, (width*0.93, 20))
            
            # Update scale
            scale = self.context['scale']
            scale_text = "Scale: %.2f" % scale
            scale_img = font.render(scale_text, True, WHITE)
            screen.blit(scale_img, (width*0.93, 40))
            
            # Update speed
            speed = dt/dt_default
            speed_text = "Speed: %.2f" % speed
            speed_img = font.render(speed_text, True, WHITE)
            screen.blit(speed_img, (width*0.93, 60))
            
            # Update display
            pygame.display.flip()
            clock.tick(fps)
            
        pygame.quit()        