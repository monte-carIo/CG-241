import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import math

def divide_triangle(a, b, c, n, res):
    if n > 0:
        v1 = (a + b)/2
        v1 = v1 / np.linalg.norm(v1)
        v2 = (a + c)/2
        v2 = v2 / np.linalg.norm(v2)
        v3 = (b + c)/2
        v3 = v3 / np.linalg.norm(v3)
        divide_triangle(a, v1, v2, n-1, res)
        divide_triangle(v1, b, v3, n-1, res)
        divide_triangle(v2, c, v3, n-1, res)
        divide_triangle(v1, v2, v3, n-1, res)
    else:
        res.extend([a,b,c])

def create_sphere():
    r = 1
    vertices = []
    indices = []
    init_vertices = np.array([
        [0, 1, 0],
        [-math.sqrt(6)/3, -1/3, math.sqrt(2)/3],
        [math.sqrt(6)/3, -1/3, math.sqrt(2)/3],
        [0, -1/3, -2*math.sqrt(2)/3],
    ])
    vertices = init_vertices
    res = []
    n = 6
    divide_triangle(init_vertices[0], init_vertices[1], init_vertices[2], n, res)
    divide_triangle(init_vertices[0], init_vertices[1], init_vertices[3], n, res)
    divide_triangle(init_vertices[0], init_vertices[2], init_vertices[3], n, res)
    divide_triangle(init_vertices[1], init_vertices[2], init_vertices[3], n, res)
    vertices = np.array(res, dtype=np.float32) 
    indices = [i for i in range(len(vertices))]
    
    return vertices, np.array(indices, dtype=np.uint32)  

class Sphere_rec:
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
        GL.glDrawElements(GL.GL_TRIANGLES,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
