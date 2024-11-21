from libs.buffer import *
from libs import transform as T
from libs.shader import *
from libs.utils import *
import ctypes
import glfw
import math
import numpy as np
from sympy import *

class CameraObject:
    def __init__(self, view_matrix, vert_shader, frag_shader):
        """
        Initialize the Camera object.
        :param view_matrix: 4x4 matrix representing the camera's view matrix.
        :param vert_shader: Vertex shader path.
        :param frag_shader: Fragment shader path.
        """
        self.view_matrix = view_matrix
        self.position = self.extract_position()
        self.axes = self.extract_axes()

        # Create the vertices for the axes
        self.vertices, self.colors, self.indices = self.create_axes_geometry()

        # OpenGL setup
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def extract_position(self):
        """Extract the camera's position from the view matrix."""
        inv_view = np.linalg.inv(self.view_matrix)
        return inv_view[:3, 3]  # Translation is the 4th column of the inverse view matrix

    def extract_axes(self):
        """Extract the camera's Right, Up, and Forward vectors."""
        inv_view = np.linalg.inv(self.view_matrix)
        right = inv_view[:3, 0]  # Right vector
        up = inv_view[:3, 1]     # Up vector
        forward = -inv_view[:3, 2]  # Forward vector (negated)
        return right, up, forward

    def create_axes_geometry(self):
        """
        Create geometry for the axes (Right, Up, Forward) as lines.
        Each axis originates from the camera's position.
        """
        right, up, forward = self.axes
        vertices = [
            *self.position, *(self.position + right),    # Right axis
            *self.position, *(self.position + up),       # Up axis
            *self.position, *(self.position + forward)   # Forward axis
        ]

        colors = [
            [1, 0, 0], [1, 0, 0],  # Red for Right axis
            [0, 1, 0], [0, 1, 0],  # Green for Up axis
            [0, 0, 1], [0, 0, 1]   # Blue for Forward axis
        ]

        indices = [
            0, 1,  # Right axis
            2, 3,  # Up axis
            4, 5   # Forward axis
        ]

        return np.array(vertices, dtype=np.float32), np.array(colors, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def setup(self):
        """Setup OpenGL buffers and shaders."""
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        projection = T.perspective(45, 1, 0.1, 100)  # Example projection matrix
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)

        return self

    def draw(self, projection, view, model):
        """Render the camera's axes."""
        GL.glUseProgram(self.shader.render_idx)

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(view, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_LINES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    # def update_view_matrix(self, new_view_matrix):
    #     """Update the camera's view matrix and recalculate geometry."""
    #     self.view_matrix = new_view_matrix
    #     self.position = self.extract_position()
    #     self.axes = self.extract_axes()
    #     self.vertices, _, _ = self.create_axes_geometry()
    #     self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
