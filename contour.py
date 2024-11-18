from libs.buffer import *
from libs import transform as T
from libs.shader import *
from libs.utils import *
import ctypes
import glfw
import math
import numpy as np
from sympy import *

class Contour(object):
    def __init__(self, points, vert_shader, frag_shader, num_levels=10):
        self.vertices, self.indices, self.colors = self.generate_contours(points, num_levels)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def generate_contours(self, points, num_levels):
        # Assuming points are structured as a grid (rows x cols)
        rows = int(math.sqrt(len(points)))
        cols = rows
        z_values = [vertex[2] for vertex in points]
        min_z, max_z = min(z_values), max(z_values)
        interval = (max_z - min_z) / num_levels if max_z > min_z else 1

        contour_lines = []
        colors = []
        for level in range(num_levels):
            z_level = min_z + level * interval
            for j in range(rows - 1):
                for i in range(cols - 1):
                    # Get vertices of the cell (4 corners)
                    idx = j * cols + i
                    cell = [
                        points[idx],
                        points[idx + 1],
                        points[idx + cols],
                        points[idx + cols + 1]
                    ]
                    # Process edges of the cell
                    edges = [
                        (cell[0], cell[1]),  # Top edge
                        (cell[0], cell[2]),  # Left edge
                        (cell[1], cell[3]),  # Right edge
                        (cell[2], cell[3])   # Bottom edge
                    ]
                    for p1, p2 in edges:
                        if (p1[2] - z_level) * (p2[2] - z_level) < 0:
                            # Interpolate
                            t = (z_level - p1[2]) / (p2[2] - p1[2])
                            contour_point = [
                                p1[0] + t * (p2[0] - p1[0]),
                                p1[1] + t * (p2[1] - p1[1]),
                                z_level
                            ]
                            contour_lines.append(contour_point)
                            # Assign color based on the level
                            color = [
                                level / (num_levels - 1),  # R
                                1 - level / (num_levels - 1),  # G
                                0  # B
                            ]
                            colors.append(color)

        vertices = np.array(contour_lines, dtype=np.float32)
        indices = np.arange(len(vertices), dtype=np.uint32)
        colors = np.array(colors, dtype=np.float32)

        return vertices, indices, colors

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        self.vao.add_ebo(self.indices)

        projection = T.ortho(-0.5, 2.5, -0.5, 1.5, -1, 1)
        modelview = np.identity(4, 'f')

        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_LINES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):
        if key == glfw.KEY_UP:
            # Handle increasing contour levels or other interactions
            pass
        if key == glfw.KEY_DOWN:
            # Handle decreasing contour levels or other interactions
            pass
