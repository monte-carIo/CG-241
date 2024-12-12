import OpenGL.GL as GL  # standard Python OpenGL wrapper
import numpy as np
from libs.shader import *
from libs import transform as T
from libs.buffer import *

def generate_plane():
    # Plane vertices (XY-plane)
    vertices = [
        [-0.5, -0.5, 0.0],  # Bottom-left
        [0.5, -0.5, 0.0],   # Bottom-right
        [0.5,  0.5, 0.0],   # Top-right
        [-0.5,  0.5, 0.0]   # Top-left
    ]

    # Plane indices: Two triangles forming a rectangle
    indices = [
        0, 1, 2,  # First triangle (bottom-left, bottom-right, top-right)
        0, 2, 3   # Second triangle (bottom-left, top-right, top-left)
    ]

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

class Mesh:
    def __init__(self, normals=None, colors=None, vert_shader=None, frag_shader=None):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        
        # Use provided normals or generate placeholder normals
        if normals is None:
            normals = np.zeros_like(self.vertices)
        self.normals = np.array(normals, dtype=np.float32)

        # Use provided colors or default to white
        if colors is None:
            colors = np.ones((len(self.vertices), 3), dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)

        self.vao = VAO()

        # Compile shaders if provided, or leave uninitialized
        if vert_shader and frag_shader:
            self.shader = Shader(vert_shader, frag_shader)
            self.uma = UManager(self.shader)
        else:
            self.shader = None
            self.uma = None

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        self.vao.add_ebo(self.indices)

        if self.shader:
            GL.glUseProgram(self.shader.render_idx)
            projection = T.ortho(-1, 1, -1, 1, -1, 1)
            modelview = np.identity(4, dtype=np.float32)

            self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
            self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        modelview = view @ model
        
        if self.shader:
            self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
            self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
            self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)
            GL.glUseProgram(self.shader.render_idx)
        
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
