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

class Object:
    
    def __init__(self, name, position, velocity, mass, color, density):
        self.name =  name
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.color = color
        self.density = density
        self.radius = self.get_radius()
        self.screen_position = None
        self.z = None
    
    def get_radius(self):
        volume = self.mass/self.density
        radius = ((3*volume)/(4*math.pi))**(1/3)
        return radius
    
    
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
        observer_params = observer_params or {"position":[0.0, 0.0, 1/scale], "theta":0.0, "phi":0.0}
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
        self.dragging = False
        
    def get_screen_position(self, obj):
        p0 = obj.position
        p_obs = self.observer.position
        p1 = p0 + p_obs
        p2 = F.rotation_3d(p1, self.observer.theta, self.observer.phi)
        p3 = p2*self.scale
        p = p3[:2] + [self.width/2,self.height/2]
        z = p3[2]
        return p, z

    def draw(self, obj):
        d = F.cartesian_distance(obj.position, self.observer.position)
        r = max(math.atan(obj.radius/d)*self.scale*obj.radius, 1.0)
        pygame.draw.circle(self.canvas, obj.color, obj.screen_position.astype(int), r)

    def update_info_text(self):
        # Update FPS
        effective_fps = self.clock.get_fps()
        fps_text = "FPS: %.0f" % effective_fps
        fps_img = self.font.render(fps_text, True, WHITE)
        self.screen.blit(fps_img, (self.width*0.80, 20))

    def handle_user_input(self, event):
        if event.type == pygame.QUIT:
            print("Quitting...")
            self.running = False
        elif event.type == pygame.MOUSEWHEEL:
            # Wheel up zooms in 5%
            if event.y > 0:
                self.observer.position 
                self.scale  *= 1.05**abs(event.y) 
            # Wheel down zooms out 5%
            elif event.y < 0:
                self.scale  *= 0.95**abs(event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Left click drags view
            if event.button == 1:
                self.dragging = True
                self.mouse_x, self.mouse_y = event.pos 
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
                self.observer.theta += dtheta
                self.observer.phi -= dphi
                self.mouse_x = mouse_x0
                self.mouse_y = mouse_y0


    def get_objects(self, state):
        objects = []
        for i in range(state["n"]):
            properties = {
                "name": state["names"][i],
                "position": np.array(state["positions"][i]),
                "velocity": np.array(state["velocities"][i]),
                "mass": state["masses"][i],
                "color": state["colors"][i],
                "density": state["densities"][i]

            }
            obj = Object(**properties)
            q, z = self.get_screen_position(obj)
            obj.screen_position = q
            obj.z = z
            objects.append(obj)
        objects.sort(key=lambda x: x.z)
        return objects

        
    def play(self):
        pygame.init()
        pygame.display.set_caption('cosmosim 0.10')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode([self.width, self.height])
        self.screen.set_alpha(None)
        self.canvas = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont(None, 24)
        self.running = True
        self.frame = 0
        while self.running:
            state = self.states[self.frame] 
            objects = self.get_objects(state)
            self.canvas.fill(BLACK)
            for obj in objects:
                self.draw(obj)
            self.screen.blit(self.canvas, [0.0,0.0])
            # Handle user inputs
            for event in pygame.event.get():
                self.handle_user_input(event)
            # Update info text
            self.update_info_text()
            # Flip display
            pygame.display.flip()
            # Update clock
            self.clock.tick(self.fps)
            # Quit on final frame, else go to next frame
            if self.frame > self.frames:
                self.running = False
            else:
                self.frame += 1
        pygame.quit()




            
        
    

    

          
class InteractiveAnimation:
    
    def __init__(self, data, width=1600, height=1000, fps=60, scale=1.3e-6):
        self.width = width
        self.height = height
        self.fps = fps
        self.default_scale = scale
        self.scale = scale        
        self.paused = False
        self.dragging = False
        self.playback_tracking = False
        self.origin_offset = np.array([self.width/2,self.height/2])
        self.context = {
            "scale":self.scale,
            "offset":np.array([0.0,0.0]),
            "rotation":np.array([0.0, 0.0]),
            "origin": self.origin_offset
        }
        self.current_state = None
        self.selected_object = None
        self.locked_object = None
        self.selected_object_name = None
        self.tracked_objects = {}
        # initialize buttons
        self.close_btn = None
        self.track_btn = None
        self.lock_btn = None
        self.playback_tracker = None
        self.pauseplay_btn = None
        
        if isinstance(data, str):
            self.states = []
            path = data
            self.filelist = os.listdir(path)
            for i in tqdm(self.filelist, desc="Loading data"):
                with open(path + i, 'r') as f:
                    #self.states.append(*json.load(f))
                    states = json_unzip(json.load(f))
                    self.states += states
        else:
            self.states = data
            
        self.frames = len(self.states)
        self.states.sort(key=lambda x: x["iterations"])

    def onscreen(self, coordinates, z):
        x, y = coordinates
        onscreen = (x >= 0 and x <= self.width and y >= 0 and y <= self.height)
        return onscreen
                    
    def draw(self, state):
        for field in ["masses","densities","positions"]:
            state[field] = np.array(state[field])
        Q = self.screen_positions
        Z = self.Z
        scale = self.context["scale"]
        names = state["names"]
        colors = state["colors"]
        radii = self.radii
        objects = sorted(zip(Q, Z, names, colors, radii), key=lambda x: x[1])
        for q, z, name, color, r in objects:
            # Trace the object's path if it is tracked
            if name in self.tracked_objects.keys():
                history = self.tracked_objects[name]
                self.draw_history(history, color)
            # Ensure object is on the screen
            if self.onscreen(q,z):
                # Ensure object is always visible
                radius = max(1, r*scale)
                # Highlight the object if selected
                if self.selected_object_name == name:
                    pygame.draw.circle(self.canvas, WHITE, q.astype(int), max(int(radius*2), 4)) 
                    pygame.draw.circle(self.canvas, BLACK, q.astype(int), max(int(radius*1.85), 2))
                # Draw the object
                pygame.draw.circle(self.canvas, color, q.astype(int), radius)
        self.screen.blit(self.canvas, [0.0,0.0])

    def draw_history(self, history, color):
        h_length = len(history)
        history_sc = F.screen_coordinates_3d_multi(np.array(history), **self.context)[0]
        for _i in range(h_length):
            if _i > 0:
                q0 = history_sc[_i-1]
                q1 = history_sc[_i]
                alpha = _i/h_length
                trail_color = tuple(alpha*c for c in color)
                pygame.draw.line(self.canvas, trail_color, q0.astype(int), q1.astype(int), 1)
                
    def toggle_tracking(self, obj):
        if obj in self.tracked_objects.keys():
            del self.tracked_objects[obj]
        else:
            i = self.current_state["names"].index(obj)
            p = self.current_state["positions"][i]
            self.tracked_objects[obj] = [p]
            
    def toggle_locking(self, obj):
        if self.locked_object == obj:
            self.locked_object = None
        else:
            self.locked_object = obj
            
    def handle_user_input(self, event):
        obj = self.selected_object
        state = self.current_state
        pos = pygame.mouse.get_pos()
        # Stop simulation when user quits
        if event.type == pygame.QUIT:
            print("Quitting...")
            self.restart = True
            self.running = False
        # Key presses
        elif event.type == pygame.KEYDOWN:
            # Spacebar pauses
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            # WASD moves the screen
            elif event.key == pygame.K_w:
                self.context['offset'] += np.array([0.0,50.0])/self.context['scale']
            elif event.key == pygame.K_a:
                self.context['offset'] += np.array([50.0,0.0])/self.context['scale']
            elif event.key == pygame.K_s:
                self.context['offset'] += np.array([0.0,-50.0])/self.context['scale']
            elif event.key == pygame.K_d:
                self.context['offset'] += np.array([-50.0,0.0])/self.context['scale']
            # R restarts the simulation
            elif event.key == pygame.K_r:
                self.restart = True
            else:
                pass
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
                if self.playback_tracker and self.playback_tracker.collidepoint(pos):
                    self.playback_tracking = True
                    self.mouse_x, self.mouse_y = event.pos
                else:
                    self.dragging = True
                    self.mouse_x, self.mouse_y = event.pos 
            # Right click resets view
            elif event.button == 3:
                self.context['offset']  = np.array([0.0,0.0])
                self.context['scale']  = self.default_scale
                self.context['rotation'] = np.array([0.0,0.0])
                self.context['origin'] = self.origin_offset
                self.locked_object = None
        elif event.type == pygame.MOUSEBUTTONUP:
            # Releasing left click
            if event.button == 1:            
                self.dragging = False
                self.playback_tracking = False
                # Click on close button
                if self.close_btn and self.close_btn.collidepoint(pos):
                    self.selected_object = None
                    self.selected_object_name = None
                # Click on object tracking button
                elif self.track_btn and self.track_btn.collidepoint(pos) and obj != None:
                    self.toggle_tracking(state["names"][obj])
                # CLick on lock button
                elif self.lock_btn and self.lock_btn.collidepoint(pos) and obj != None:
                    self.toggle_locking(state["names"][obj])
                # Click on pause/play button
                elif self.pauseplay_btn_rect and self.pauseplay_btn_rect.collidepoint(pos):
                    self.paused = not self.paused
                # Click on planet
                else:
                    selected_objs = [i for i, p in enumerate(self.screen_positions) if np.linalg.norm(p - pos) <= max(self.radii[i]*self.context['scale'],10.0)]
                    if len(selected_objs) > 0:
                        self.selected_object = selected_objs[0]
                        self.selected_object_name = state["names"][self.selected_object]

        elif event.type == pygame.MOUSEMOTION:
            # Dragging mouse rotates the screen
            if self.dragging:
                mouse_x0, mouse_y0 = event.pos
                dtheta = (mouse_x0 - self.mouse_x)/self.width
                dphi = (mouse_y0 - self.mouse_y)/self.height
                self.context['rotation'] += np.array([dtheta,dphi])
                self.mouse_x = mouse_x0
                self.mouse_y = mouse_y0
            elif (
                    self.playback_tracking 
                    and self.iterations > 0 
                    and self.iterations < self.frames
                    #and self.playback_tracker.collidepoint(event.pos)
                ):
                mouse_x0, mouse_y0 = event.pos
                pct_of_bar_moved = (mouse_x0 - self.mouse_x)/(self.width*0.8)
                frames_moved = int(self.frames*pct_of_bar_moved)
                self.iterations = min(max(self.iterations + frames_moved, 0), self.frames)
                self.mouse_x = mouse_x0
                self.mouse_y = mouse_y0
        else:
            pass

    def update_playback_controls(self):
        # Playback bar
        bar_length = self.width*0.8
        bar_start = self.width*0.1
        bar_end = self.width*0.9
        bar_height = self.height*0.95
        pygame.draw.line(self.screen, WHITE, (bar_start, bar_height), (bar_end, bar_height), 5)
        # Playback tracker
        progress = self.iterations/self.frames
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
            
    def update_simulation_text(self):
        # Update FPS
        effective_fps = self.clock.get_fps()
        fps_text = "FPS: %.0f" % effective_fps
        fps_img = self.font.render(fps_text, True, WHITE)
        self.screen.blit(fps_img, (self.width*0.80, 20))
        # Update scale
        scale = self.context['scale']
        scale_text = "Scale: {:.2e}".format(scale)
        scale_img = self.font.render(scale_text, True, WHITE)
        self.screen.blit(scale_img, (self.width*0.80, 40))
        # Update iterations
        iterations = self.iterations
        frames = self.frames
        iterations_text = f"Frames: {iterations}/{frames} ({iterations/frames:.0%})"
        iterations_img = self.font.render(iterations_text, True, WHITE)
        self.screen.blit(iterations_img, (self.width*0.80, 60))
        # Update elapsed time
        elapsed_time = iterations*self.current_state.get("dt",1)
        elapsed_time_formatted = str(datetime.timedelta(seconds=elapsed_time))
        elapsed_time_text = f"Elapsed time: {elapsed_time_formatted}"
        elapsed_time_img = self.font.render(elapsed_time_text, True, WHITE)
        self.screen.blit(elapsed_time_img, (self.width*0.80, 80))
        # Paused text
        if self.paused:
            paused_text = "PAUSED"
            paused_img = self.font.render(paused_text, True, WHITE)
            self.screen.blit(paused_img, (self.width*0.48, 20))

    def update_selected_object_text(self):
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
                #for state in self.states:
                while self.iterations < self.frames:
                    state = self.states[self.iterations]
                    self.current_state = state
                    new_state = True
                    for field in ["masses","densities","positions"]:
                        self.current_state[field] = np.array(state[field])
                   
                    while self.running and not self.restart and (self.paused or new_state):
                        # Refresh state
                        state = self.states[self.iterations]
                        self.current_state = state
                        # Refresh radii
                        self.radii = [F.get_radius(m,d) for m, d in zip(state["masses"], state["densities"])]
                        # Update selected object index
                        if self.selected_object_name in state["names"]:
                            self.selected_object = state["names"].index(self.selected_object_name)
                        else:
                            self.selected_object = None
                            self.selected_object_name = None
                        # Update locked object
                        if self.locked_object not in state["names"]:
                            self.locked_object = None
                        # Update tracked object index and positions
                        abs_obj = []
                        for obj in self.tracked_objects.keys():
                            if obj in state["names"]:
                                i = state["names"].index(obj)
                                p = state["positions"][i]
                                self.tracked_objects[obj].append(p)
                                self.tracked_objects[obj] = self.tracked_objects[obj][-1000:]
                            else:
                                abs_obj.append(obj)
                        for obj in abs_obj:
                            del self.tracked_objects[obj]
                        # Determine offset
                        if self.locked_object != None:
                            obj_position = state["positions"][state["names"].index(self.locked_object)]
                            self.context["offset"] = np.array([0.0,0.0])
                            new_offset = (self.context["origin"] - F.screen_coordinates_3d(obj_position, **self.context)[0])
                            self.context["offset"] = new_offset/self.context["scale"]
                        # Clear the screen
                        self.screen.fill(BLACK)
                        self.canvas.fill(BLACK)
                        # Update screen positions
                        self.screen_positions, self.Z = F.screen_coordinates_3d_multi(state["positions"], **self.context)
                        # Draw
                        self.draw(state)
                        # Update simulation text
                        self.update_simulation_text()
                        # Update selected object text
                        self.update_selected_object_text()
                        # Update playback controls
                        self.update_playback_controls()
                        # Handle user inputs
                        for event in pygame.event.get():
                            self.handle_user_input(event)
                        # Refresh display
                        pygame.display.flip()
                        self.clock.tick(self.fps)
                        new_state = False
                    self.iterations += 1
                self.iterations = 0
                self.clear_tracking_histories()
            self.restart = False
        pygame.quit()

    def clear_tracking_histories(self):
        for obj in self.tracked_objects:
            self.tracked_objects[obj] = []