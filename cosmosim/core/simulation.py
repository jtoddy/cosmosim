from cosmosim.core.new import UniverseState
import cosmosim.util.functions as F
import math
import os
import pickle
import pygame
import numpy as np
import copy
from tqdm import tqdm

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


class Simulation:
    
    def __init__(self, objects, dt, iterations, outpath=None, filesize=1000):
        self.objects = objects
        self.dt = dt
        self.iterations = iterations
        self.outpath = outpath
        self.filesize = filesize
               
    def run(self):
        state = UniverseState(self.objects)
        nfiles = math.ceil(self.iterations/self.filesize)
        elapsed = 0
        if self.outpath:
            if not os.path.isdir(self.outpath):
                os.mkdir(self.outpath)
            for n in range(nfiles): 
                path = self.outpath + f"{n}.dat"
                for i in tqdm(range(self.filesize), desc=f"Writing file {n}"):
                    state.interact(self.dt)
                    with open(path, "ab+") as f:
                        state.save(f)
                    elapsed += 1
                    if elapsed >= self.iterations:
                        break
        else:
            states = []
            for i in tqdm(range(self.iterations), desc="Running simulation"):
                state.interact(self.dt)
                new_state = copy.deepcopy(state)
                states.append(new_state)
            return states
                
class Animation:
    
    def __init__(self, data, width=1600, height=1000, fps=60, scale=1.3e-6):
        self.width = width
        self.height = height
        self.fps = fps
        self.default_scale = scale
        self.scale = scale        
        self.paused = False
        self.dragging = False
        self.context = {
            "scale":self.scale,
            "offset":np.array([0.0,0.0]),
            "rotation":np.array([0.0, 0.0]),
            "origin":np.array([self.width/2,self.height/2])
        }
        
        if isinstance(data, str):
            self.states = []
            path = data
            self.filelist = os.listdir(path)
            for i in tqdm(self.filelist, desc="Loading data"):
                with open(path + i, 'rb') as f:
                    while True:
                        try:
                            state = pickle.load(f)
                            self.states.append(state)
                        except EOFError:
                            break
        else:
            self.states = data
            
        self.frames = len(self.states)
                    
    def draw(self, state):
        objects = state.existing_objects()
        scale = self.context['scale']
        for obj in objects:
            radius = max(1, int(obj.get_radius()*scale))
            q = F.screen_coordinates_3d(obj.position, **self.context)
            pygame.draw.circle(self.screen, obj.color, q.astype(int), radius)
            
    def handle_user_input(self, event):
        # Stop simulation when user quits
        if event.type == pygame.QUIT:
            print("Quitting...")
            self.running = False
        # Key presses
        elif event.type == pygame.KEYDOWN:
            # Spacebar pauses
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            # WASD moves the screen
            elif event.key == pygame.K_w:
                self.context['offset'] += np.array([0,50])/self.context['scale']
            elif event.key == pygame.K_a:
                self.context['offset'] += np.array([50,0])/self.context['scale']
            elif event.key == pygame.K_s:
                self.context['offset'] += np.array([0,-50])/self.context['scale']
            elif event.key == pygame.K_d:
                self.context['offset'] += np.array([-50,00])/self.context['scale']
        elif event.type == pygame.MOUSEWHEEL:
            # Wheel up zooms in 5%
            if event.y > 0:
                self.context['scale']  *= 1.05**abs(event.y) 
            # Wheel down zooms out 5%
            elif event.y < 0:
                self.context['scale']  *= 0.95**abs(event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Left click drags view
            if event.button == 1:
                self.dragging = True
                self.mouse_x, self.mouse_y = event.pos 
            # Right click resets view
            elif event.button == 3:
                self.context['offset']  = np.array([0.0,0.0])
                self.context['scale']  = self.default_scale
                self.context['rotation'] = np.array([0.0,0.0])
        elif event.type == pygame.MOUSEBUTTONUP:
            # Releasing left click
            if event.button == 1:            
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            # Dragging mouse rotates the screen
            if self.dragging:
                mouse_x0, mouse_y0 = event.pos
                dtheta = (mouse_x0 - self.mouse_x)/self.width
                dphi = (mouse_y0 - self.mouse_y)/self.height
                self.context['rotation'] += np.array([dtheta,dphi])
        else:
            pass
            
    def update_simulation_text(self):
        # Update FPS
        P = (self.clock.get_time() + 1)/1000
        effective_fps = 1/P
        fps_text = "FPS: %.0f" % effective_fps
        fps_img = self.font.render(fps_text, True, WHITE)
        self.screen.blit(fps_img, (self.width*0.90, 20))
        # Update scale
        scale = self.context['scale']
        scale_text = "Scale: {:.2e}".format(scale)
        scale_img = self.font.render(scale_text, True, WHITE)
        self.screen.blit(scale_img, (self.width*0.90, 40))
        # Update iterations
        iterations = self.iterations
        frames = self.frames
        iterations_text = f"Iterations: {iterations}/{frames}"
        iterations_img = self.font.render(iterations_text, True, WHITE)
        self.screen.blit(iterations_img, (self.width*0.90, 60))
        # Paused text
        if self.paused:
            paused_text = "PAUSED"
            paused_img = self.font.render(paused_text, True, WHITE)
            self.screen.blit(paused_img, (self.width*0.48, 20))
            
    def play(self, paused=False):
        pygame.init()
        pygame.display.set_caption('cosmosim 0.10')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode([self.width, self.height])
        self.font = pygame.font.SysFont(None, 24)
        self.running = True
        self.iterations = 0
        self.paused = paused
        while self.running:
            for state in self.states:
                new_state = True
                while self.running and (self.paused or new_state):
                    # Clear the screen
                    self.screen.fill(BLACK)
                    # Handle user inputs
                    for event in pygame.event.get():
                        self.handle_user_input(event)
                    # Draw
                    self.draw(state)
                    # Update simulation text
                    self.update_simulation_text()
                    # Refresh display
                    pygame.display.flip()
                    self.clock.tick(self.fps)
                    new_state = False
                self.iterations += 1
            self.running = False
        pygame.quit()
        