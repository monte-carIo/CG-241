import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean windows system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
from pywavefront import Wavefront  # for OBJ file loading
from libs.transform import *
from itertools import cycle
from object import Cat


class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """
    def __init__(self, width=480 * 2, height=480):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        self.trackball = Trackball()
        self.mouse = (0, 0)

        glfw.make_context_current(self.win)
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_window_size_callback(self.win, self.on_resize)
        glfw.set_window_refresh_callback(self.win, self.on_idle)

        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.drawables = []

    def run(self):
        self.win_size = glfw.get_window_size(self.win)
        angle = 0
        while not glfw.window_should_close(self.win):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)

            GL.glViewport(0, 0, int(self.win_size[0] / 2), int(self.win_size[1]))
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix((int(self.win_size[0] / 2), int(self.win_size[1])))
            model = np.identity(4, 'f')

            for drawable in self.drawables:
                drawable.draw(projection, view, model)

            GL.glViewport(int(self.win_size[0] / 2), 0, int(self.win_size[0] / 2), int(self.win_size[1]))
            for drawable in self.drawables:
                drawable.draw(projection, view, model)

            glfw.swap_buffers(self.win)
            glfw.poll_events()

    def add(self, *drawables):
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        if action in {glfw.PRESS, glfw.REPEAT}:
            if key in {glfw.KEY_ESCAPE, glfw.KEY_Q}:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

    def on_mouse_move(self, win, xpos, ypos):
        old_pos = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old_pos, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old_pos, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])

    def on_resize(self, win, width, height):
        self.win_size = (width, height)
        GL.glViewport(0, 0, width, height)

    def on_idle(self, _win):
        pass



def main():
    viewer = Viewer()
    obj_model = Cat('obj\\cat\\20430_Cat_v1_NEW.obj', 'obj\\cat\\20430_cat_diff_v1.jpg',
                    "triangle\gouraud.vert", "triangle\gouraud.frag").setup()
    viewer.add(obj_model)
    viewer.run()


if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()
