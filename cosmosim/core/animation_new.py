import cosmosim.util.functions as F
import os
import sys
import pickle
import pygame
import numpy as np
import datetime
import json
import math
from cosmosim.util.constants import WHITE, BLACK, ME, AU
from cosmosim.util.json_zip import json_unzip
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
     
class Observer:
    
    def __init__(self, position, theta=0.0, phi=0.0):
        self.position = position
        self.theta = theta
        self.phi = phi


class Animation:
    
    def __init__(self, data, width=1600, height=1000, fps=60, scale=1.3e-6, observer_params=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.scale = scale
        observer_params = observer_params or {"position":[0.0, 0.0, 0.0], "theta":0.0, "phi":0.0}
        self.observer = Observer(**observer_params)
        self.states = self.load_data(data)
        self.frames = len(self.states)
        self.initialize_ui()
        
    def load_data(self, data):
        if isinstance(data, str):
            states = []
            path = data
            filelist = os.listdir(path)
            for i in tqdm(filelist, desc="Loading data"):
                with open(path + i, 'r') as f:
                    states += json_unzip(json.load(f))
        else:
            states = data
        states.sort(key=lambda x: x["iterations"])
        return states
    
    def initialize_ui(self):
        self.paused = False
        self.dragging = False
        self.playback_tracking = False
        self.playback_tracker = None
        self.pauseplay_btn = None
        self.pauseplay_btn_rect = None
        self.track_btn = None
        self.lock_btn = None

        
    def get_screen_positions(self, positions):
        p = np.array(positions)
        disp_scale = np.array([self.width/self.height, 1, 1])
        disp_pixels = np.array([self.width, self.height, 0])
        # Convert to observer coordinates
        R = np.matmul(F.Ry(self.observer.theta), F.Rx(self.observer.phi))
        p0 = R.dot(p.T).T - self.observer.position
        z = (-p0[:,2]).copy()
        p1 = (p0/z[None,:].T)#[:,[0,1]]
        p2 = (p1*2 + disp_scale)/(2*disp_scale)
        q = p2*disp_pixels*np.array([1, -1, 1]) + np.array([0, self.height, 0])
        q[:,2] = z
        return q
    
    def onscreen(self, screen_coordinates, r):
        x, y, z = screen_coordinates
        onscreen = (x + r >= 0) and (x - r <= self.width) and (y + r >= 0) and (y - r <= self.height) and (z > 0)
        return onscreen

    def draw(self, i, state):
        q = state["screen_positions"][i]
        r0 = state["radii"][i]
        color = state["colors"][i]
        r = max(self.height*(r0/q[2]), 1.0)
        if self.onscreen(q, r):
            pygame.draw.circle(self.canvas, color, q[:2], r)
   
    def handle_user_input(self, event):
        if event.type == pygame.QUIT:
            print("Quitting...")
            self.running = False
        elif event.type == pygame.MOUSEWHEEL:
            # Wheel up zooms in 5%
            if event.y > 0:
                self.observer.position = np.array(self.observer.position) - np.array([0.0, 0.0, self.observer.position[2]*(0.05*abs(event.y))])
            # Wheel down zooms out 5%
            elif event.y < 0:
                self.observer.position = np.array(self.observer.position) + np.array([0.0, 0.0, self.observer.position[2]*(0.05*abs(event.y))])
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.playback_tracker and self.playback_tracker.collidepoint(event.pos):
                self.playback_tracking = True
                self.mouse_x, self.mouse_y = event.pos
            # Left click drags view
            elif event.button == 1:
                self.dragging = True
                self.mouse_x, self.mouse_y = event.pos 
        elif event.type == pygame.MOUSEBUTTONUP:
            # Releasing left click
            if event.button == 1:            
                self.dragging = False
                self.playback_tracking = False
                # Click on pause/play button
                if self.pauseplay_btn_rect and self.pauseplay_btn_rect.collidepoint(event.pos):
                    self.paused = not self.paused
                # Click on planet
                else:
                    selected_objs = [i for i, p in enumerate(self.current_state["screen_positions"]) if np.linalg.norm(p[:,[1,2]] - event.pos) <= max(self.height*(self.current_state["radii"][i]/self.current_state["screen_positions"][i][2]),10.0)]
                    if len(selected_objs) > 0:
                        self.selected_object = selected_objs[-1]
                        self.selected_object_name = self.current_state["names"][self.selected_object]
        elif event.type == pygame.MOUSEMOTION:
            # Dragging mouse rotates the screen
            if self.dragging:
                mouse_x0, mouse_y0 = event.pos
                dtheta = (mouse_x0 - self.mouse_x)/self.width
                dphi = (mouse_y0 - self.mouse_y)/self.height
                self.observer.theta -= dtheta
                self.observer.phi -= dphi
                self.mouse_x = mouse_x0
                self.mouse_y = mouse_y0
            elif (
                    self.playback_tracking 
                    and self.frame > 0 
                    and self.frame < self.frames
                    #and self.playback_tracker.collidepoint(event.pos)
                ):
                mouse_x0, mouse_y0 = event.pos
                pct_of_bar_moved = (mouse_x0 - self.mouse_x)/(self.width*0.8)
                frames_moved = int(self.frames*pct_of_bar_moved)
                self.frame = min(max(self.frame + frames_moved, 0), self.frames)
                self.mouse_x = mouse_x0
                self.mouse_y = mouse_y0
    
    def render_info_text(self):
        # Update FPS
        effective_fps = max(self.clock.get_fps(), 1)
        fps_text = "FPS: %.0f" % effective_fps
        fps_img = self.font.render(fps_text, True, WHITE)
        self.screen.blit(fps_img, (self.width*0.80, 20))
        # Update frames
        frame = self.frame
        frames = self.frames
        frames_text = f"Frames: {frame}/{frames} ({frame/frames:.0%})"
        frames_img = self.font.render(frames_text, True, WHITE)
        self.screen.blit(frames_img, (self.width*0.80, 40))
        # Update elapsed time
        elapsed_time = round(frame*self.current_state.get("dt",1)/self.fps)
        elapsed_time_formatted = str(datetime.timedelta(seconds=elapsed_time))
        elapsed_time_text = f"Elapsed time: {elapsed_time_formatted}"
        elapsed_time_img = self.font.render(elapsed_time_text, True, WHITE)
        self.screen.blit(elapsed_time_img, (self.width*0.80, 60))
        # Paused text
        if self.paused:
            paused_text = "PAUSED"
            paused_img = self.font.render(paused_text, True, WHITE)
            self.screen.blit(paused_img, (self.width*0.48, 20))

    def render_selected_object_text(self):
        obj = self.selected_object
        state = self.current_state
        if obj != None:
            if state["names"][obj]:
                name = state["names"][obj]
                # Info box dimensions
                INFO_X = 10
                INFO_Y = 10
                INFO_WIDTH = 420
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
                if self.selected_object_name in self.tracked_objects.keys():                  
                    self.track_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X,INFO_Y2),(INFO_WIDTH/4,BTN_SIZE)), 0)
                    track_img = self.font.render("TRACK", True, BLACK)
                else:
                    self.track_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X,INFO_Y2),(INFO_WIDTH/4,BTN_SIZE)), 2)
                    track_img = self.font.render("TRACK", True, WHITE)
                self.screen.blit(track_img, (INFO_X+TXT_PAD, INFO_Y2+TXT_PAD-1))
                
                # Lock button
                if self.selected_object_name == self.locked_object:                  
                    self.lock_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X+(INFO_WIDTH/4),INFO_Y2),(INFO_WIDTH/4,BTN_SIZE)), 0)
                    lock_img = self.font.render("LOCK", True, BLACK)
                else:
                    self.lock_btn = pygame.draw.rect(self.screen, WHITE, ((INFO_X+(INFO_WIDTH/4),INFO_Y2),(INFO_WIDTH/4,BTN_SIZE)), 2)
                    lock_img = self.font.render("LOCK", True, WHITE)
                self.screen.blit(lock_img, (INFO_X+TXT_PAD+(INFO_WIDTH/4), INFO_Y2+TXT_PAD-1))
                
                # Planet info text
                p = state["positions"][obj]
                v = state["velocities"][obj]
                q = self.screen_positions[obj]
                z = self.Z[obj]
                vmag = np.linalg.norm(np.array(v))
                ptext1 = name.upper()
                ptext2 = "Mass: {:.2e} Earth masses".format(state["masses"][obj]/ME)
                ptext3 = "Radius: {:.2e} km".format(self.radii[obj]/1000)
                ptext4 = "Position: [{:.1e}, {:.1e}, {:.1e}] m".format(*p)
                ptext5 = "Velocity: [{:.1e}, {:.1e}, {:.1e}] {:.1e} m/s".format(*v, vmag)
                ptext6 = "Screen coordinates: [{:.0f}, {:.0f}] z={:.1e}".format(*q, z)
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
                self.selected_object = None
    
    def render_playback_controls(self):
        # Playback bar
        bar_length = self.width*0.8
        bar_start = self.width*0.1
        bar_end = self.width*0.9
        bar_height = self.height*0.95
        pygame.draw.line(self.screen, WHITE, (bar_start, bar_height), (bar_end, bar_height), 5)
        # Playback tracker
        progress = (self.frame+1)/self.frames
        tracker_x = bar_start + min(max((bar_length*progress),0), bar_length)
        tracker_y = bar_height - 10
        self.playback_tracker = pygame.draw.line(self.screen, WHITE, (tracker_x, tracker_y), (tracker_x, tracker_y+20), 8)
        # Pause/play button
        btn_size = 25
        if self.paused:
            self.pauseplay_btn = pygame.image.load("cosmosim/resources/play_icon.png")
        else:
            self.pauseplay_btn = pygame.image.load("cosmosim/resources/pause_icon.png")
        self.pauseplay_btn = pygame.transform.scale(self.pauseplay_btn, (btn_size, btn_size))
        self.pauseplay_btn_rect = self.pauseplay_btn.get_rect()
        self.pauseplay_btn_rect = self.pauseplay_btn_rect.move(bar_end+15, bar_height-btn_size/2)
        self.screen.blit(self.pauseplay_btn, (bar_end+15, bar_height-btn_size/2))
        # Runtime progress
        elapsed_time = round(self.frame/self.fps)
        total_time = round(self.frames/self.fps)
        elapsed_time_formatted = str(datetime.timedelta(seconds=elapsed_time))
        total_time_formatted = str(datetime.timedelta(seconds=total_time))
        elapsed_time_text = f"{elapsed_time_formatted}/{total_time_formatted}"
        elapsed_time_img = self.font.render(elapsed_time_text, True, WHITE)
        self.screen.blit(elapsed_time_img, (20, bar_height-(elapsed_time_img.get_height()/3)))

    def draw_objects(self):
        state = self.current_state
        state["screen_positions"] = self.get_screen_positions(state["positions"])
        state["radii"] = [F.get_radius(m,d) for m, d in zip(state["masses"], state["densities"])]
        plot_order = np.flip(state["screen_positions"][:,2].argsort())
        for i in plot_order:
            self.draw(i, state)
        self.screen.blit(self.canvas, [0.0,0.0])

    def draw_ui(self):
        # Update info text
        self.render_info_text()
        # Render playback controls
        self.render_playback_controls()

        
    def play(self, paused=False):
        pygame.init()
        pygame.display.set_caption('cosmosim 0.10')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode([self.width, self.height])
        self.screen.set_alpha(None)
        self.canvas = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont(None, 24)
        self.running = True
        self.paused = paused
        self.frame = 0
        self.current_state = None
        self.selected_object = None
        self.selected_object_name = None
        while self.running:
            self.canvas.fill(BLACK)
            self.current_state = self.states[self.frame]
            # Draw objects
            self.draw_objects()
            # Draw UI
            self.draw_ui()
            # Handle user inputs
            for event in pygame.event.get():
                self.handle_user_input(event)
            # Flip display
            pygame.display.flip()
            # Update clock
            self.clock.tick(self.fps)
            # Quit on final frame, else go to next frame
            if self.paused:
                pass
            elif self.frame >= self.frames-1:
                self.frame = 0
            else:
                self.frame += 1
        pygame.quit()