from libs.buffer import *
from libs import transform as T
from libs.shader import *
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

    def key_handler(self, key):
        """Handle key events for the camera."""
        pass
    
class MasterCamera(CameraObject):
    def __init__(self, view_matrix, vert_shader, frag_shader, fov=45, aspect=1, near=0.1, far=1, color=[0, 1, 0]):
        """
        Initialize the Master Camera object with a frustum visualization.
        :param view_matrix: 4x4 matrix representing the camera's view matrix.
        :param vert_shader: Vertex shader path.
        :param frag_shader: Fragment shader path.
        :param fov: Field of view in degrees.
        :param aspect: Aspect ratio (width/height).
        :param near: Distance to the near plane.
        :param far: Distance to the far plane.
        """
        super().__init__(view_matrix, vert_shader, frag_shader)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        # Create frustum geometry
        self.frustum_vertices, self.frustum_colors, self.frustum_indices = self.create_frustum_geometry(color)

    def create_frustum_geometry(self, color):
        """
        Create geometry for the camera's frustum.
        """
        h_near = 2 * np.tan(np.radians(self.fov) / 2) * self.near
        w_near = h_near * self.aspect
        h_far = 2 * np.tan(np.radians(self.fov) / 2) * self.far
        w_far = h_far * self.aspect

        # Define frustum vertices relative to the camera's local space
        near_center = self.position + self.axes[2] * self.near  # Position + Forward * near
        far_center = self.position + self.axes[2] * self.far   # Position + Forward * far

        # Near plane corners
        near_top_left = near_center + self.axes[1] * (h_near / 2) - self.axes[0] * (w_near / 2)
        near_top_right = near_center + self.axes[1] * (h_near / 2) + self.axes[0] * (w_near / 2)
        near_bottom_left = near_center - self.axes[1] * (h_near / 2) - self.axes[0] * (w_near / 2)
        near_bottom_right = near_center - self.axes[1] * (h_near / 2) + self.axes[0] * (w_near / 2)

        # Far plane corners
        far_top_left = far_center + self.axes[1] * (h_far / 2) - self.axes[0] * (w_far / 2)
        far_top_right = far_center + self.axes[1] * (h_far / 2) + self.axes[0] * (w_far / 2)
        far_bottom_left = far_center - self.axes[1] * (h_far / 2) - self.axes[0] * (w_far / 2)
        far_bottom_right = far_center - self.axes[1] * (h_far / 2) + self.axes[0] * (w_far / 2)

        # Combine all vertices
        vertices = [
            *near_top_left, *near_top_right, *near_bottom_left, *near_bottom_right,
            *far_top_left, *far_top_right, *far_bottom_left, *far_bottom_right
        ]

        # Color the frustum lines (e.g., green for all lines)
        colors = [color for _ in range(8)]  # Green for frustum

        # Indices to form lines
        indices = [
            0, 1, 1, 3, 3, 2, 2, 0,  # Near plane edges
            4, 5, 5, 7, 7, 6, 6, 4,  # Far plane edges
            0, 4, 1, 5, 2, 6, 3, 7   # Connecting edges
        ]

        return np.array(vertices, dtype=np.float32), np.array(colors, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def setup(self):
        """Setup OpenGL buffers and shaders for the Master Camera."""
        super().setup()  # Call the base class setup

        # Add buffers for the frustum geometry
        self.vao.add_vbo(0, self.frustum_vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.frustum_colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_ebo(self.frustum_indices)

        return self

    def draw(self, projection, view, model):
        """Render the Master Camera with its frustum."""
        super().draw(projection, view, model)  # Draw the base camera geometry

        GL.glUseProgram(self.shader.render_idx)

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(view, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_LINES, self.frustum_indices.shape[0], GL.GL_UNSIGNED_INT, None)
