# ------------ Package Imports ------------

import OpenGL.GL as GL
import glfw
import numpy as np
from time import sleep
from SGD import evaluate_points, gradient_descent, generate_points

# ------------ Library Imports ------------

from libs.transform import *
from itertools import cycle

# ------------ Shape Imports --------------

from mesh import *
from linear import *
from sphere import *


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

        # Register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        # Display OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode(), 
              'GLSL', GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode(),
              'Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # Set up initial GL state
        GL.glClearColor(0.5, 0.5, 0.5, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        # Initialize drawables and parameters
        self.drawables = []
        self.gradient = []
        self.points_line = []
        self.radius = 0

    def run(self):
        """Main render loop for the OpenGL window with 2D contour overlay and top-down view."""
        frame_count = 0
        max_frames = int(len(self.points_line) * 4 / 5)
        
        while not glfw.window_should_close(self.win):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Window dimensions for setting up viewports
            win_size = glfw.get_window_size(self.win)
            main_viewport_size = win_size  # Full window
            top_viewport_size = (int(win_size[0] * 0.3), int(win_size[1] * 0.3))  # Bottom-left viewport size (30% of window)

            # ---- Main 3D View ----
            GL.glViewport(0, 0, *main_viewport_size)

            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            # Calculate transformation matrices
            frame_count %= max_frames
            normal_vec = normalized(vec(-self.gradient[frame_count][0], -self.gradient[frame_count][1], 1))
            trans_matrix = translate(vec(self.points_line[frame_count]) + self.radius * normal_vec)
            self.rot_matrix = self.calculate_rotation_matrix(frame_count, normal_vec)
            model = trans_matrix @ self.rot_matrix

            # Draw 3D objects in main view
            for drawable in self.drawables:
                drawable.draw(projection, view, model)

            # ---- Fixed Top View ----
            GL.glViewport(0, 0, *top_viewport_size)  # Set bottom-left viewport

            # Define top-view projection (orthographic) and view matrix (top-down)
            top_projection = ortho(-1, 1, -1, 1, 0.1, 100)
            top_view = look_at([0, 0, 10], [0, 0, 0], [0, 1, 0])  # Looking from above along Z-axis

            for drawable in self.drawables:
                drawable.draw(top_projection, top_view, model)

            # Swap buffers and poll events
            glfw.swap_buffers(self.win)
            glfw.poll_events()
            frame_count += 1
            sleep(0.008)


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
        if action in {glfw.PRESS, glfw.REPEAT}:
            if key in {glfw.KEY_ESCAPE, glfw.KEY_Q}:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
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
    
    mesh_points = generate_points(15, 70)
    evaluated_points, mean_z = evaluate_points(mesh_points, expression_str)

    plane = Mesh(evaluated_points, mean_z, 'resources/shaders/phong_texture.vert', 'resources/shaders/phong_texture.frag').setup()
    viewer.points_line, viewer.gradient = gradient_descent(0.01, expression_str)
    # breakpoint()
    line = Linear(viewer.points_line, 'resources/shaders/phong.vert', 'resources/shaders/phong.frag').setup()
    sphere = Sphere(viewer.radius, 'resources/shaders/phong_texture.vert', 'resources/shaders/phong_texture.frag').setup()

    viewer.add(plane, line, sphere)
    viewer.run()
    glfw.terminate()


if __name__ == '__main__':
    main()
