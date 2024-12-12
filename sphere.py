import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import math

def create_sphere():
    r = 0.5
    vertices = []
    indices = []
    for phi in np.linspace(-math.pi/2, math.pi/2, 40):
        for theta in np.linspace(0, 2 * math.pi, 40):
            
            x = r * math.sin(theta) * math.cos(phi)
            z = r * math.cos(theta) * math.cos(phi)
            y = r * math.sin(phi)
            vertices.append([x,y,z])
    
    for i in range(39):
        for j in range(39):
            indices.append(i*40 + j)
            indices.append(i*40 + j + 1)
            indices.append((i+1)*40 + j)
            indices.append((i+1)*40 + j + 1)
            indices.append(i*40 + j + 1)
    
    return vertices, np.array(indices, dtype=np.uint32)  

class Sphere:
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices = create_sphere()
        self.vertices = np.array(self.vertices, dtype=np.float32)
        normals = np.random.normal(0, 3, (len(self.vertices), 3))
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.colors = np.zeros((len(self.vertices), 3)).astype(np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        
        self.vao.add_ebo(self.indices)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        # modelview = np.identity(4, 'f')
        modelview = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)

        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        modelview = view @ model
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
