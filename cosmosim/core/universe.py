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
        self.angular_momentum = 0.0
        self.total_energy = 0.0
        self.clicked_planet = None
        self.running = False
        self.paused = False
        self.dragging = False
        self.iterations = 0
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
        
    def random_planet(self, density=1, min_mass=1, max_mass=100, dmin=0, 
                      dmax=500, min_velocity=1.0, max_velocity=3.0, 
                      name=None, color=None):
        mass = random.randint(min_mass, max_mass)
        radius = F.get_radius(mass,density)
        ppolar = [random.randint(dmin,dmax), random.random()*(2*math.pi)]
        position = F.to_cartesian(*ppolar)
        v0 = min_velocity + (max_velocity - min_velocity) * random.random()
        v_theta = 2*math.pi * random.random()
        velocity = [v0*math.cos(v_theta),v0*math.sin(v_theta)]
        return self.create_planet(mass=mass,radius=radius,position=position,
                                  velocity=velocity,color=color,name=name)
    
    def get_active_planets(self):
        return [planet for planet in self.planets if planet.alive]
    
    def get_absorbed_planets(self):
        return [planet for planet in self.planets if not planet.alive]
    
    def get_bound_planets(self):
        return [planet for planet in self.get_active_planets() if planet.energy < 0]

    def get_unbound_planets(self):
        return [planet for planet in self.get_active_planets() if planet.energy >= 0]
    
    def update_planet_lists(self):
        self.active_planets = self.get_active_planets()
        self.absorbed_planets = self.get_absorbed_planets()
        self.bound_planets = self.get_bound_planets()
        self.unbound_planets = self.get_unbound_planets()
        
    def handle_user_input(self, event):
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
                self.dt *= 1.05
            if event.key == pygame.K_KP_MINUS:
                self.dt *= 0.95
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Left click drags view
            if event.button == 1:
                self.dragging = True
                self.mouse_x, self.mouse_y = event.pos 
            # Right click resets view
            elif event.button == 3:
                self.context['offset']  = np.array([0.0,0.0])
                self.context['scale']  = 1.0
                self.dt = dt_default
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
                    clicked_planets = [p for p in self.get_active_planets() if np.linalg.norm(F.screen_coordinates(p.position, **self.context) - pos) <= p.radius*self.context['scale']]
                    if len(clicked_planets) > 0:
                        if self.clicked_planet:
                            self.clicked_planet.clicked = False
                        self.clicked_planet = clicked_planets[0]
                        self.clicked_planet.clicked = True
        elif event.type == pygame.MOUSEMOTION:
            # Dragging mouse moves the screen
            if self.dragging:
                mouse_x0, mouse_y0 = event.pos
                dx = int((mouse_x0 - self.mouse_x)/(5*self.context['scale']))
                dy = int((mouse_y0 - self.mouse_y)/(5*self.context['scale']))
                self.context['offset'] += np.array([dx,dy])
        else:
            pass
    
    def interact(self):
        # Calculations
        if not self.paused:
            # Initial conditions
            m = np.array([p.mass for p in self.active_planets])
            p0 = np.array([p.position for p in self.active_planets])
            v0 = np.array([p.velocity for p in self.active_planets])
            # Calculate net accelerations
            a_3d = acc_blas(p0, m, self.G)  # Magic!!!
            a = np.delete(a_3d, 2, 1)       # Only need 2D acceleration
            # Integration
            v = v0 + a*self.dt
            p = p0 + v*self.dt
            # Calculate energies
            d_raw = pairwise_distances(p, n_jobs=-1)
            d = np.clip(d_raw, 1e-9, None) # Clip to avoid overflows
            d_inv = np.reciprocal(d, where=(d!=0))
            v_mag = np.linalg.norm(v, axis=1)
            G = -self.G*np.multiply.outer(m,m)
            U = np.sum(G*d_inv, axis=1)
            K = 0.5*m*(v_mag**2)
            self.total_energy = np.sum(K + U)
            # Calculate angular momenta
            self.angular_momentum = np.sum(m*np.cross(p,v))
            # Identify collisions
            if self.collisions:
                radii = np.array([p.radius for p in self.active_planets])
                collision_matrix = d <= np.add.outer(radii,radii)
        # Update planets
        for i, planet in enumerate(self.active_planets):
            if not self.paused:
                if planet.alive:
                    if not planet.immobile:
                        planet.velocity = v[i]
                        planet.position = p[i]
                    if self.collisions:
                        for j, c in enumerate(collision_matrix[i]):
                            if c and i != j:
                                other_planet = self.active_planets[j]
                                if planet.mass >= other_planet.mass:
                                    planet.absorb(other_planet)
                                else:
                                    other_planet.absorb(planet)
                    planet.energy =  K[i] + U[i]
                planet.update_history(self.trail_length)
        
    def draw_planets(self):
        for planet in self.active_planets:
            planet.draw(self.screen, planet.color)
            
    def update_universe_info_text(self):
        universe_text = [
            "Total Planets: " + str(len(self.active_planets)), 
            "Bound Planets: " + str(len(self.bound_planets)),
            "Escaped Planets: " + str(len(self.unbound_planets)),
            "Destroyed Planets: " + str(len(self.absorbed_planets)),
            "Net Angular Momentum: {:.2e}".format(self.angular_momentum),
            "Total Energy: {:.2e}".format(self.total_energy)
        ]
        
        for i, text in enumerate(universe_text):
            img = self.font.render(text, True, WHITE)
            self.screen.blit(img, (20, (20)*(i+1)))
            
    def update_clicked_planet_text(self):
        if self.clicked_planet:
            if self.clicked_planet.alive:
                # Info box dimensions
                INFO_X = 10
                INFO_Y = 150
                INFO_WIDTH = 240
                INFO_HEIGHT = 150
                BTN_SIZE = 30
                TXT_PAD = 10
                INFO_X2 = INFO_X + INFO_WIDTH
                INFO_Y2 = INFO_Y + INFO_HEIGHT
                
                # Box Outline
                pygame.draw.rect(self.screen, WHITE, ((INFO_X, INFO_Y), (INFO_WIDTH, INFO_HEIGHT)), 2)
                
                # Header bar
                pygame.draw.line(self.screen, WHITE, (INFO_X,INFO_Y+BTN_SIZE), (INFO_X2,INFO_Y+BTN_SIZE), 1)
                
                # Close button
                self.close_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X2-BTN_SIZE,INFO_Y),(BTN_SIZE,BTN_SIZE)), 2)
                close_img = self.font.render("X", True, WHITE)
                self.screen.blit(close_img, (INFO_X2-BTN_SIZE+TXT_PAD,INFO_Y+TXT_PAD))
                
                # Track button
                if self.clicked_planet.tracked:                  
                    self.track_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X,INFO_Y2),(INFO_WIDTH/2,BTN_SIZE)), 0)
                    track_img = self.font.render("TRACK", True, BLACK)
                else:
                    self.track_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X,INFO_Y2),(INFO_WIDTH/2,BTN_SIZE)), 2)
                    track_img = self.font.render("TRACK", True, WHITE)
                self.screen.blit(track_img, (INFO_X+TXT_PAD, INFO_Y2+TXT_PAD-1))
                
                # Destroy button
                self.destroy_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X+(INFO_WIDTH/2),INFO_Y2),(INFO_WIDTH/2,BTN_SIZE)), 2)
                destroy_img = self.font.render("DESTROY", True, WHITE)
                self.screen.blit(destroy_img, (INFO_X+(INFO_WIDTH/2)+TXT_PAD, INFO_Y2+TXT_PAD-1))
                
                # Planet info text
                ptext1 = self.clicked_planet.name.upper()
                ptext2 = "Mass: %.0f" % round(self.clicked_planet.mass,2) 
                ptext3 = "Radius: %.2f" % self.clicked_planet.radius
                ptext4 = "Velocity: [%.2f %.2f] %.2f" % (*self.clicked_planet.velocity, np.linalg.norm(self.clicked_planet.velocity))
                ptext5 = "Energy: {:.2e}".format(round(self.clicked_planet.energy, 2))
                ptext6 = "Planets eaten: %i" % self.clicked_planet.planets_eaten
                pimg1 = self.font.render(ptext1, True, WHITE)
                pimg2 = self.font.render(ptext2, True, WHITE)
                pimg3 = self.font.render(ptext3, True, WHITE) 
                pimg4 = self.font.render(ptext4, True, WHITE)
                pimg5 = self.font.render(ptext5, True, WHITE)
                pimg6 = self.font.render(ptext6, True, WHITE)
                self.screen.blit(pimg1, (20, INFO_Y + TXT_PAD))
                self.screen.blit(pimg2, (20, INFO_Y + 2*TXT_PAD + 20))
                self.screen.blit(pimg3, (20, INFO_Y + 2*TXT_PAD + 40))
                self.screen.blit(pimg4, (20, INFO_Y + 2*TXT_PAD + 60))
                self.screen.blit(pimg5, (20, INFO_Y + 2*TXT_PAD + 80))
                self.screen.blit(pimg6, (20, INFO_Y + 2*TXT_PAD + 100))
            else:
                self.clicked_planet.clicked = False
                self.clicked_planet = None
                    
    def update_simulation_text(self):
        # Update FPS
        P = (self.clock.get_time() + 1)/1000
        effective_fps = 1/P
        fps_text = "FPS: %.0f" % effective_fps
        fps_img = self.font.render(fps_text, True, WHITE)
        self.screen.blit(fps_img, (self.width*0.93, 20))
        # Update scale
        scale = self.context['scale']
        scale_text = "Scale: %.2f" % scale
        scale_img = self.font.render(scale_text, True, WHITE)
        self.screen.blit(scale_img, (self.width*0.93, 40))
        # Update speed
        speed = self.dt/self.dt_default
        speed_text = "Speed: %.2f" % speed
        speed_img = self.font.render(speed_text, True, WHITE)
        self.screen.blit(speed_img, (self.width*0.93, 60))
    
    def simulate(self, width=1600, height=1000, speed=1, fps=33, 
                 trail_length=1000, collisions=True, 
                 track_all=False, run_while=(lambda x: True)):
        # Configure simulation
        self.dt_default = speed/20
        self.dt = self.dt_default
        self.collisions = collisions
        self.trail_length = trail_length
        self.width = width
        self.height = height
        if track_all:
            for p in self.planets:
                p.tracked = True
        # Initialize display context
        self.context = {
            "scale":1.0,
            "offset":np.array([0.0,0.0]),
            "origin":np.array([width/2,height/2])
        }
        # Set up display
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.screen = pygame.display.set_mode([width, height])
        pygame.display.set_caption('cosmosim 0.10')
        # Run simulation
        self.running = True
        while self.running and run_while(self):
            # Clear the screen
            self.screen.fill(BLACK)
            # Update planet lists
            self.update_planet_lists()
            # Handle user inputs
            for event in pygame.event.get():
                self.handle_user_input(event)
            # Simulate gravitational interactions between planets
            self.interact()
            # Draw planets
            self.draw_planets()
            # Update universe info text
            self.update_universe_info_text()
            # Update clicked planet text
            self.update_clicked_planet_text()
            # Update simulation text
            self.update_simulation_text()
            # Refresh display
            pygame.display.flip()
            self.clock.tick(fps)
            # Update iterations
            self.iterations += 1
        pygame.quit()        