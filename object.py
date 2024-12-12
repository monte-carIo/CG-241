import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np
import pywavefront
from libs.shader import *
from libs import transform as T
from libs.buffer import *
import ctypes
import math 
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *

def load_obj(filename):
    vertices = []
    textures = []
    normals = []
    indices = []
    tex_indices = []
    norm_indices = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                # Parse vertex positions
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vt '):
                # Parse texture coordinates
                textures.append(list(map(float, line.strip().split()[1:3])))
            elif line.startswith('vn '):
                # Parse vertex normals
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                # Parse face indices
                face = line.strip().split()[1:]
                quad = [list(map(lambda x: int(x) if x!='' else -1, vertex.split('/'))) for vertex in face]
                # [tex_indices.append(vertex[1] - 1) for vertex in quad]
                if len(quad) == 3:
                    triangles = [[quad[0], quad[1], quad[2]]]
                else:
                    triangles = [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
                for triangle in triangles:
                    for v, vt, vn in triangle:
                        indices.append(vertices[v - 1])      # Vertex index
                        if vt != -1:
                            tex_indices.append(textures[vt - 1])  # Texture index
                        norm_indices.append(normals[vn - 1])  # Normal index

    return indices, tex_indices, norm_indices

class Cat:
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.texcoord, self.normals = load_obj('obj\\cat\\20430_Cat_v1_NEW.obj')
        # self.vertices, self.texcoord, self.normals = load_obj('obj\\61-obj\\Five_Wheeler-(Wavefront OBJ).obj')

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.normals = np.array(self.normals, dtype=np.float32)
        self.texcoord = np.array(self.texcoord, dtype=np.float32)
        
        self.model = np.array([
            [1, 0, 0, 0],
            [0, np.cos(-math.pi/2), -np.sin(-math.pi/2), -5],
            [0, np.sin(-math.pi/2), np.cos(-math.pi/2), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.colors = np.random.rand(self.vertices.shape[0], 3).astype(np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoord, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        
        # self.vao.add_ebo(self.indices)
        

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
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
        self.uma.setup_texture('tex1', 'obj\\cat\\20430_cat_diff_v1.jpg')
        # self.uma.setup_texture('tex1', 'obj\99-futuristic-five-wheeled-building-five-wheeler_2_textures\Control_Module_Basic_color-.jpg')
        # self.uma.setup_texture('tex1', 'sliver.jpg')
        
        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model, range=None):
        self.vao.activate()
        # self.uma.setup_texture('tex1', 'obj\\cat\\20430_cat_diff_v1.jpg')
        modelview = view @ model
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)
        if range is not None:
            self.uma.upload_uniform_scalar1f(range[0], 'max_v')
            self.uma.upload_uniform_scalar1f(range[1], 'min_v')
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])

class Car:
    def __init__(self, vert_shader, frag_shader):
        # self.vertices, self.texcoord, self.normals = load_obj('obj\\cat\\20430_Cat_v1_NEW.obj')
        self.vertices, self.texcoord, self.normals = load_obj('obj\\61-obj\\Five_Wheeler-(Wavefront OBJ).obj')
        self.model = np.identity(4, 'f')

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.normals = np.array(self.normals, dtype=np.float32)
        self.texcoord = np.array(self.texcoord, dtype=np.float32)

        self.colors = np.random.rand(self.vertices.shape[0], 3).astype(np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoord, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        
        # self.vao.add_ebo(self.indices)
        

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
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
        # self.uma.setup_texture('tex1', 'obj\\cat\\20430_cat_diff_v1.jpg')
        self.uma.setup_texture('tex1', 'obj\99-futuristic-five-wheeled-building-five-wheeler_2_textures\Control_Module_Basic_color-.jpg')
        # self.uma.setup_texture('tex1', 'sliver.jpg')
        
        shininess = 100.0
        mode = 1

        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        return self

    def draw(self, projection, view, model, range=None):
        self.vao.activate()
        # self.uma.setup_texture('tex1', 'obj\99-futuristic-five-wheeled-building-five-wheeler_2_textures\Control_Module_Basic_color-.jpg')
        modelview = view @ model
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)
        if range is not None:
            self.uma.upload_uniform_scalar1f(range[0], 'max_v')
            self.uma.upload_uniform_scalar1f(range[1], 'min_v')
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])