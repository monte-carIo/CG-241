# ------------ Package Imports ------------
import OpenGL.GL as GL
import glfw
import numpy as np
import random
from copy import deepcopy
from time import sleep
from sympy import symbols, lambdify
import torch
import imgui
from imgui.integrations.glfw import GlfwRenderer

# ------------ Library Imports ------------
from libs.transform import *
from itertools import cycle

# ------------ Shape Imports --------------
from mesh import *
from linear import *
from sphere import *
from contour import *

from SGD import evaluate_points, gradient_descent, generate_points
from models.MLP import ComplexMLP, get_point, get_trajectory

def generate_cameras(top_view_matrix):
    # Define 45-degree rotation matrices for x and y axes
    cos_45 = np.sqrt(2) / 2
    sin_45 = np.sqrt(2) / 2

    # 45-degree rotation around the x-axis
    rotation_x_pos = np.array([
        [1, 0, 0, 0],
        [0, cos_45, -sin_45, 0],
        [0, sin_45, cos_45, 0],
        [0, 0, 0, 1]
    ])
    
    rotation_x_neg = np.array([
        [1, 0, 0, 0],
        [0, cos_45, sin_45, 0],
        [0, -sin_45, cos_45, 0],
        [0, 0, 0, 1]
    ])
    
    # 45-degree rotation around the y-axis
    rotation_y_pos = np.array([
        [cos_45, 0, sin_45, 0],
        [0, 1, 0, 0],
        [-sin_45, 0, cos_45, 0],
        [0, 0, 0, 1]
    ])
    
    rotation_y_neg = np.array([
        [cos_45, 0, -sin_45, 0],
        [0, 1, 0, 0],
        [sin_45, 0, cos_45, 0],
        [0, 0, 0, 1]
    ])
    
    flip_yx = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    flip_z_pos = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    flip_z_neg = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    flip_z_135_pos = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    flip_z_45_neg = np.array([
        [cos_45, sin_45, 0, 0],
        [-sin_45, cos_45, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    flip_z_45_pos = np.array([
        [cos_45, -sin_45, 0, 0],
        [sin_45, cos_45, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Apply rotations to the top view matrix to create the 6 different views
    views = [
        top_view_matrix,                                  # Top view
        flip_yx @ rotation_x_pos @ top_view_matrix,                 # Positive 45 around x-axis
        rotation_x_neg @ top_view_matrix,                 # Negative 45 around x-axis
        flip_z_pos @ rotation_y_pos @ top_view_matrix,                 # Positive 45 around y-axis
        flip_z_neg @ rotation_y_neg @ top_view_matrix,                 # Negative 45 around y-axis
        flip_z_pos @ flip_z_45_pos @ rotation_x_pos @ rotation_y_pos @ top_view_matrix,  # Positive 45 around x and y axes
        flip_z_45_neg @ rotation_x_neg @ rotation_y_neg @ top_view_matrix   # Negative 45 around x and y axes
    ]
    
    corrected_views = []
    original_position = np.array([3, 0, -30, 1])
    for view in views:
        # Extract the rotation part of the matrix
        rotation_part = view[:3, :3]
        
        # Calculate the new position by applying the rotation to the original position vector
        corrected_position = original_position[:3]
        
        # Create a new matrix with the corrected translation
        corrected_view = view.copy()
        corrected_view[:3, 3] = corrected_position
        
        corrected_views.append(corrected_view)

    return corrected_views

class Viewer:
    def __init__(self, width=1600, height=800):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # OpenGL window settings
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        
        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        glfw.make_context_current(self.win)

        # Initialize trackball and mouse
        self.trackball = Trackball()
        self.mouse = (0, 0)
        
        imgui.create_context()
        self.impl = GlfwRenderer(self.win)

        # Register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        # Display OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode(), 
              'GLSL', GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode(),
              'Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # Set up initial GL state
        GL.glClearColor(0., 0., 0., 0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        # Initialize drawables and parameters
        self.model = None
        self.rotating = False
        self.debug = False
        self.drawables = []
        self.gradient = []
        self.points_line = []
        self.radius = 0
        self.generate_cams = True
        self.cams = []
        self.cam_index = -1
        self.change_size = None
        self.update = None
        self.reset_state = None
        self.selected_item = None
        self.model_dict = {}
        self.model_dict['Optimizer'] = {
                    'lr 0.0x': [1,4,6,10],
                    'optimizer': ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'AdamW'],
                    'velocity': [1, 2, 10, 1]
                }
        # self.model_dict['Optimizer'] = {
        #         'optimizer': ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'AdamW']}
        self.optims = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'AdamW']
        
    def run(self):
        """Main render loop for the OpenGL window with 2D contour overlay."""
        frame_count = 0
        max_frames = int(len(self.points_line) * 4 / 5)
        self.dropdown_items = list(self.model_dict.keys())
        self.selected_item = 0  # Index for dropdown
        self.selected_optim = 1  # Index for optimizer dropdown
        self.veclocity = 2
        
        while not glfw.window_should_close(self.win):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Set up 3D viewport for drawing objects
            win_size = glfw.get_window_size(self.win)
            main_viewport_size = (int(win_size[0] * 0.5), win_size[1])
            top_viewport_size = (int(win_size[0] * 0.5), win_size[1]) 
            
            # ---- ImGui overlay ----
            # Poll for and process events
            glfw.poll_events()
            # Process ImGui inputs
            self.impl.process_inputs()
            # Start a new frame for ImGui
            imgui.new_frame()
            # Render the UI window
            self.render_ui()

            if self.update:
                current_model = self.model_dict[self.dropdown_items[self.selected_item]]
                current_optimizer = self.optims[self.selected_optim]
                self.args = {key: (value[1]) for key, value in current_model.items() if len(value) > 1}
                print(self.args, current_optimizer)
                self.update = False
                if self.veclocity != self.args['velocity']:
                    self.veclocity = self.args['velocity']
                else:
                    for drawable in self.drawables:
                        # breakpoint()
                        self.points_line, self.gradient = get_trajectory(self.model, optimizer=current_optimizer, lr= 0.01 * self.args['lr 0.0x'])
                        max_frames = int(len(self.points_line) * 4 / 5)
                        if hasattr(drawable, 'update'):
                            drawable.update(self.points_line)
            
            # ---- Main 3D View ----
            GL.glViewport(int(win_size[0] * 0.5), 0, *main_viewport_size)

            if self.cam_index == -1:
                view = self.trackball.view_matrix()
            if self.generate_cams:
                self.cams = generate_cameras(view)
                self.generate_cams = False
            projection = self.trackball.projection_matrix(main_viewport_size)
            if self.cam_index != -1:
                if self.cam_index < len(self.cams):
                    view = self.cams[self.cam_index]
                else:
                    self.cam_index = -1
            if self.debug:
                print('Projection matrix:', projection)
                print('View matrix:', view)
                self.debug = False
            # breakpoint()
            if self.rotating:
                self.trackball.rotation = quaternion_mul(self.trackball.rotation, quaternion_from_euler(0.5, 0, 0))

            # Calculate transformation matrices
            frame_count %= max_frames
            # breakpoint()
            normal_vec = normalized(vec(-self.gradient[frame_count][0], -self.gradient[frame_count][1], 1))
            trans_matrix = translate(vec(self.points_line[frame_count]) + self.radius * normal_vec)
            self.rot_matrix = self.calculate_rotation_matrix(frame_count, normal_vec)
            model = trans_matrix @ self.rot_matrix
            
            # Draw all 3D objects
            for drawable in self.drawables:
                if drawable.__class__.__name__ != 'Contour':
                    drawable.draw(projection, view, model)
                
            # ---- Fixed Top View ----
            GL.glViewport(0, 0, *top_viewport_size)   # Set bottom-left viewport

            # Define top-view projection (orthographic) and view matrix (top-down)
            top_view = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, -30],
                                [0, 0, 0, 1]], dtype=np.float32)
            top_projection = self.trackball.projection_matrix(top_viewport_size)

            for drawable in self.drawables:
                if drawable.__class__.__name__ != 'Mesh':
                    drawable.draw(top_projection, top_view, model)
            
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.win)
            glfw.poll_events()
            frame_count += 1
            if self.veclocity != 0:
                sleep(0.1/(self.veclocity ** 2))

    def render_ui(self):
        # Get the number of parameters for the selected model
        num_params = 5
        # print(num_params)
        # Calculate window height dynamically (you can adjust these values for better layout)
        base_height = 90  # Base height for the window header
        param_height = 36  # Approximate height per slider
        dynamic_height = base_height + (num_params * param_height)

        if (self.change_size):
            self.reset_state = True
            imgui.set_next_window_size(280, dynamic_height)
        
        # Start a new ImGui window for the UI
        imgui.begin("Settings", closable=False)
        
        imgui.text("Configure the viewer window")
        
        imgui.push_item_width(imgui.get_window_width() * 0.65)
        
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        
        self.change_size, self.selected_item = imgui.combo("##Models", self.selected_item, self.dropdown_items, height_in_items = 4)
        changed, self.selected_optim = imgui.combo("##Optimizers", self.selected_optim, self.optims, height_in_items = 4)
        imgui.same_line(spacing = 10)
        imgui.text("Models")
        
        if changed:
            self.update = True
        
        for item in self.model_dict[self.dropdown_items[self.selected_item]]:
            tmp = self.model_dict[self.dropdown_items[self.selected_item]][item]
            
            imgui.spacing()  # Adds vertical spacing
            imgui.spacing()
            imgui.spacing()
            if isinstance(tmp[0], int):
                temp = tmp[1]
                
                changed, tmp[1] = imgui.slider_int(f"##{item}",tmp[1], min_value = tmp[0], max_value = tmp[-2], format='%d')
                
                if str(item).endswith("point") or str(item).endswith("range") or item in ['function']:
                    if not self.update and temp != tmp[0]:
                        self.update = True
                elif item not in ['smooth']:
                    if not self.update and temp != tmp[1]:
                        self.update = True
                else:
                    if not self.update and temp != tmp[0]:
                        self.update = True

                imgui.same_line(spacing = 10)
                imgui.text(f'{item}')
        
        # End the ImGui window
        imgui.end()

    def calculate_rotation_matrix(self, frame_count, normal_vec):
        """Calculate rotation matrix based on frame count and normal vector"""
        if frame_count == 0:
            return identity()
        else:
            prev_point = self.points_line[frame_count - 1]
            curr_point = self.points_line[frame_count]
            velocity_vec = vec([curr - prev for curr, prev in zip(curr_point, prev_point)])
            distance = np.linalg.norm(velocity_vec)
            rot_axis = normalized(np.cross(normal_vec, velocity_vec + self.radius * normal_vec))
            angle = 360 * distance / (self.radius * 2 * np.pi)
            return rotate(rot_axis, angle) @ self.rot_matrix

    def add(self, *drawables):
        """Add drawable objects to the scene"""
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """Handle keyboard events for controlling the viewer"""
        nums_keys = [glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5, glfw.KEY_6, glfw.KEY_7, glfw.KEY_8, glfw.KEY_9]
        if action in {glfw.PRESS, glfw.REPEAT}:
            if key in {glfw.KEY_ESCAPE, glfw.KEY_Q}:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_R:
                self.rotating = not self.rotating
            if key == glfw.KEY_D:
                self.debug = True
            if key in nums_keys:
                self.cam_index = nums_keys.index(key)
            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)

    def on_mouse_move(self, win, xpos, ypos):
        """Handle mouse movement for rotation and panning"""
        old_pos = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old_pos, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old_pos, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """Handle scroll events for zooming"""
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])


def main():
    glfw.init()
    viewer = Viewer()
    viewer.radius = 0.4
    expression_str = input("Enter a mathematical expression for z in terms of x and y (e.g., 'sin(x) + cos(y)'): ")
    breakpoint()
    mesh_points = generate_points(15, 70)
    evaluated_points, mean_z = evaluate_points(mesh_points, expression_str)

    plane = Mesh(evaluated_points, mean_z, 'resources/shaders/gouraud.vert', 'resources/shaders/gouraud.frag').setup()
    viewer.points_line, viewer.gradient = gradient_descent(0.01, expression_str)
    line = Linear(viewer.points_line, 'resources/shaders/phong.vert', 'resources/shaders/phong.frag').setup()
    sphere = Sphere(viewer.radius, 'resources/shaders/phong_texture.vert', 'resources/shaders/phong_texture.frag').setup()

    viewer.add(plane, line, sphere)


    viewer.run()
    glfw.terminate()
    
def mainv2():
    glfw.init()
    viewer = Viewer()
    viewer.radius = 0.2
    model = ComplexMLP()
    # save the model
    # torch.save(model, 'model.pth')
    # model = torch.load('model.pth')
    model = model.eval()
    evaluated_points = get_point(model, 15, 70)
    viewer.points_line, viewer.gradient = get_trajectory(model)
    viewer.model = model
    plane = Mesh(evaluated_points, 'resources/shaders/gouraud.vert', 'resources/shaders/phong.frag').setup()
    line = Linear(viewer.points_line, 'resources/shaders/phong.vert', 'resources/shaders/phong.frag').setup()
    sphere = Sphere(viewer.radius, 'resources/shaders/phong_texture.vert', 'resources/shaders/phong_texture.frag').setup()
    # breakpoint()
    contour = Contour(evaluated_points, 'resources/shaders/contour.vert', 'resources/shaders/contour.frag').setup()

    viewer.add(plane, line, sphere, contour)

    viewer.run()
    glfw.terminate()

if __name__ == '__main__':
    mainv2()
