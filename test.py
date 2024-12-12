import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import os

def load_texture(filename):
    img = Image.open(filename)
    img_data = img.convert("RGBA").tobytes()
    width, height = img.size

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return texture_id

def load_obj(filename):
    vertices = []
    textures = []
    normals = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vt '):
                tex_coords = list(map(float, line.strip().split()[1:]))
                if len(tex_coords) >= 2:  # Ensure only the first two values are used
                    textures.append(tex_coords[:2])
            elif line.startswith('vn '):
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = []
                for vert in line.strip().split()[1:]:
                    indices = vert.split('/')
                    face.append(tuple(map(int, indices)))
                faces.append(face)

    return vertices, textures, normals, faces

def render_model(vertices, textures, normals, faces, texture_id):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex in face:
            v_idx, t_idx, n_idx = vertex
            if t_idx > 0:
                glTexCoord2fv(textures[t_idx - 1])
            if n_idx > 0:
                glNormal3fv(normals[n_idx - 1])
            glVertex3fv(vertices[v_idx - 1])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    gluPerspective(45, display[0] / display[1], 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
    obj_file = "obj\\cat\\20430_Cat_v1_NEW.obj"
    texture_file = "obj\\cat\\20430_cat_diff_v1.jpg"

    vertices, textures, normals, faces = load_obj(obj_file)
    texture_id = load_texture(texture_file)

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_model(vertices, textures, normals, faces, texture_id)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()