import cosmosim.util.functions as F
import os
import sys
import pickle
import pygame
import numpy as np
import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers

WHITE = (255,255,255)
YELLOW = (255,255,0)
BLACK = (0,0,0)
          
class InteractiveAnimation:
    
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
        self.dt = self.states[0].dt
        self.states.sort(key=lambda x: x.iterations)
                    
    def draw(self, state):
        radii = state.get_radii()
        Q = F.screen_coordinates_3d_multi(state.positions, **self.context)
        for i in range(state.n_objects):
            q = Q[i]
            if self.onscreen(q):
                radius = max(1, int(radii[i]*self.context['scale']))
                pygame.draw.circle(self.canvas, state.colors[i], q.astype(int), radius)
        self.screen.blit(self.canvas, self.context['offset'])
            
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
            # R restarts the simulation
            elif event.key == pygame.K_r:
                self.restart = True
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
        
    def onscreen(self, coordinates):
        x, y = coordinates
        return (x >= 0 and x <= self.width and y >= 0 and y <= self.height)
        
            
    def update_simulation_text(self):
        # Update FPS
        effective_fps = self.clock.get_fps()
        fps_text = "FPS: %.0f" % effective_fps
        fps_img = self.font.render(fps_text, True, WHITE)
        self.screen.blit(fps_img, (self.width*0.85, 20))
        # Update scale
        scale = self.context['scale']
        scale_text = "Scale: {:.2e}".format(scale)
        scale_img = self.font.render(scale_text, True, WHITE)
        self.screen.blit(scale_img, (self.width*0.85, 40))
        # Update iterations
        iterations = self.iterations
        frames = self.frames
        iterations_text = f"Iterations: {iterations}/{frames}"
        iterations_img = self.font.render(iterations_text, True, WHITE)
        self.screen.blit(iterations_img, (self.width*0.85, 60))
        # Update elapsed time
        elapsed_time = iterations*self.dt
        elapsed_time_formatted = str(datetime.timedelta(seconds=elapsed_time))
        elapsed_time_text = f"Elapsed time: {elapsed_time_formatted}"
        elapsed_time_img = self.font.render(elapsed_time_text, True, WHITE)
        self.screen.blit(elapsed_time_img, (self.width*0.85, 80))
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
        self.screen.set_alpha(None)
        self.canvas = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont(None, 24)
        self.running = True
        self.restart = False
        self.iterations = 0
        self.paused = paused
        while self.running:
            while not self.restart:
                for state in self.states:
                    new_state = True
                    while self.running and not self.restart and (self.paused or new_state):
                        # Clear the screen
                        self.canvas.fill(BLACK)
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
            self.iterations = 0
            self.restart = False
        pygame.quit()
        

class MP4Animation:
    
    def __init__(self, data, outpath, width=1000, height=1000, fps=60, scale=1.3e-6, n_frames=None, context={}):
        self.width = width
        self.height = height
        self.fps = fps
        self.default_scale = scale
        self.scale = scale        
        self.paused = False
        self.dragging = False
        self.outpath = outpath
        self.context = {
            "scale":self.scale,
            "offset":np.array([0.0,0.0]),
            "rotation":np.array([0.0, 0.0]),
            "origin":np.array([self.width/2,self.height/2]),
            **context
        }
        
        if isinstance(data, str):
            self.states = []
            path = data
            self.filelist = os.listdir(path)
            loaded_frames = 0
            for i in tqdm(self.filelist, desc="Loading data"):
                with open(path + i, 'rb') as f:
                    while (not n_frames or loaded_frames < n_frames):
                        try:
                            state = pickle.load(f)
                            self.states.append(state)
                            loaded_frames += 1
                        except EOFError:
                            break
        else:
            self.states = data
            
        self.frames = n_frames or len(self.states)
        self.dt = self.states[0].dt
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        figsize=(self.width*px, self.height*px, )
        self.fig = plt.figure(figsize=figsize)
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.ax = plt.axes(xlim=(0,width), ylim=(0,height))
        self.ax.set_facecolor("black")
        self.space = plt.scatter([],[],[],[])
        self.text = self.ax.text(self.width*0.85, self.height - 40, "", color="white")
    
    def init(self):
        self.space.set_offsets([])
        self.text.set_text("")
        return self.space, self.text
    
    def info_text(self, i):
        # Update scale
        scale = self.context['scale']
        scale_text = "Scale: {:.2e}".format(scale)
        # Update iterations
        iterations = i
        frames = self.frames
        iterations_text = f"Frame: {iterations}/{frames}"
        # Update elapsed time
        elapsed_time = iterations*self.dt
        elapsed_time_formatted = str(datetime.timedelta(seconds=elapsed_time))
        elapsed_time_text = f"Elapsed time: {elapsed_time_formatted}"
        
        info_text = f"""{scale_text}
{iterations_text}
{elapsed_time_text}"""

        return info_text
        
        
    def animate(self, i):
        #print(f"Rendering frame: {i+1}/{self.frames}")
        state = self.states[i]
        scale = self.context['scale']
        positions = [F.screen_coordinates_3d(p, **self.context) for p in state.positions]
        radii = [max(1, int(r*scale)) for r in state.get_radii()]
        colors = [(c[0]/255, c[1]/255, c[2]/255) for c in state.colors]
        self.space.set_offsets(positions)
        self.space.set_sizes(radii)
        self.space.set_edgecolors(colors)
        info_text = self.info_text(i)
        self.text.set_text(info_text)
        return self.space, self.text
    
    def run(self):
        writer = writers['ffmpeg']
        writer = writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        if not os.path.isdir(self.outpath):
            os.makedirs(self.outpath)
        existing_filelist = os.listdir(self.outpath)
        for f in existing_filelist:
            os.remove(self.outpath + f)
        anim = FuncAnimation(self.fig, 
                             self.animate, 
                             frames=tqdm(range(self.frames), desc="Rendering MP4", file=sys.stdout),
                             interval=20)
        print("saving")
        anim.save(self.outpath+"cosmosim.mp4")
        print("Done!")