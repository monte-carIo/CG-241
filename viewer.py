import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean windows system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
from circle import *
from line import *
from triangle.triangle import *
from libs.transform import *
from cam_object import *
from itertools import cycle
from object import Cat, Car
from PIL import Image
import imgui
from imgui.integrations.glfw import GlfwRenderer



# ------------  Viewer class & windows management ------------------------------
class Viewer:
    """ GLFW viewer windows, with classic initialization & graphics loop """
    def __init__(self, width=480*2, height=480):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        # version hints: create GL windows with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DEPTH_BITS, 24)  # Request a 24-bit depth buffer
        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        self.trackball = Trackball()
        self.mouse = (0, 0)


        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)
        imgui.create_context()
        self.impl = GlfwRenderer(self.win)

        # Register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_window_size_callback(self.win, self.on_resize)
        # Set idle callback
        glfw.set_window_refresh_callback(self.win, self.on_idle)
        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)  # Enable depth testing
        GL.glDepthFunc(GL.GL_LESS)


        # initially empty list of object to draw
        self.drawables = []
        self.depth_drawables = []
        self.rotate = False
        self.depth = False
        
        print('Generating cameras')
        top_view = np.array([[1, 0, 0, 0],
                [0, np.cos(math.pi/2), np.sin(-math.pi/2), 0],
                [0, np.sin(math.pi/2), np.cos(math.pi/2), -40],
                [0, 0, 0, 1]], dtype=np.float32)
        self.cams = []
        init_view = self.trackball.view_matrix()
        init_view[:3, 3] = [0, -10, -70]
        self.cams.extend([init_view, top_view])
        for angle in [45, 70]:
            self.cams.extend(generate_cameras(top_view, angle))
        self.drawables.append(MasterCamera(self.cams[0],'triangle/cam.vert',
                                           'triangle/cam.frag',
                                           color=[1,0,0]).setup())
        for cam in self.cams[1:]:
            self.drawables.append(MasterCamera(cam,'triangle/cam.vert',
                                               'triangle/cam.frag').setup())
        self.cam_index = -1
        self.save = False
    
    def two_viewports(self, angle):
        # clear draw buffer
        width, height = int(self.win_size[0]), int(self.win_size[1])
        GL.glViewport(0, 0, width // 2, height)            
        # draw our scene objects
        view = self.trackball.view_matrix()
        if self.cam_index != -1:
            if self.cam_index < len(self.cams):
                view = self.cams[self.cam_index]
            else:
                self.cam_index = -1
        projection = self.trackball.projection_matrix((width // 2, height))
        model = np.array([
            [1, 0, 0, 0],
            [0, np.cos(-math.pi/2), -np.sin(-math.pi/2), -5],
            [0, np.sin(-math.pi/2), np.cos(-math.pi/2), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        if self.rotate:
            angle += 0.001
        rotate = np.array([
            [np.cos(angle), np.sin(angle), 0, 0],
            [-np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])            
        model = model @ rotate
        for drawable in self.drawables:
            drawable.draw(projection, view, model)
        # depth viewport
        GL.glViewport(width // 2, 0, width // 2, height)
        
        for drawable in self.depth_drawables:
            drawable.draw(projection, view, model)
        
        if self.depth and self.save:
            color_buffer = GL.glReadPixels(width // 2 , 0, width // 2, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            color_array = np.frombuffer(color_buffer, dtype=np.uint8).reshape((height, width // 2, 3))
            color_array = np.flipud(color_array)
            Image.fromarray(color_array).save('color_frame.png')
            self.save = False
        
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        # flush render commands, and swap draw buffers
        glfw.swap_buffers(self.win)
        # Poll for and process events
        glfw.poll_events()
    
    def one_viewport(self, angle):
        # clear draw buffer
        width, height = int(self.win_size[0]), int(self.win_size[1])
        GL.glViewport(0, 0, width, height)            
        # draw our scene objects
        view = self.trackball.view_matrix()
        if self.cam_index != -1:
            if self.cam_index < len(self.cams):
                view = self.cams[self.cam_index]
            else:
                self.cam_index = -1
        projection = self.trackball.projection_matrix((width, height))
        model = np.array([
            [1, 0, 0, 0],
            [0, np.cos(-math.pi/2), -np.sin(-math.pi/2), -5],
            [0, np.sin(-math.pi/2), np.cos(-math.pi/2), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        if self.rotate:
            angle += 0.001
        rotate = np.array([
            [np.cos(angle), np.sin(angle), 0, 0],
            [-np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])            
        model = model @ rotate
        for drawable in self.drawables:
            drawable.draw(projection, view, model)
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        # flush render commands, and swap draw buffers
        glfw.swap_buffers(self.win)
        # Poll for and process events
        glfw.poll_events()
        
    def run(self):
        """ Main render loop for this OpenGL windows """
        self.win_size = glfw.get_window_size(self.win)
        angle = 0
        while not glfw.window_should_close(self.win):
            # ---- ImGui overlay ----
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            # Process ImGui inputs
            # Poll for and process events
            glfw.poll_events()
            self.impl.process_inputs()
            # Start a new frame for ImGui
            imgui.new_frame()
            # Render the UI window
            self.render_ui()
            if self.depth:
                self.two_viewports(angle)
            else:
                self.one_viewport(angle)

    def render_ui(self):
        # Set the size and position of the window
        imgui.set_next_window_size(350, 250)
        imgui.set_next_window_position(10, 10, imgui.ALWAYS)

        # Begin the UI window
        imgui.begin("Keyboard Controls:", closable=False)

        # Add a header with styling
        imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.6, 0.9, 1.0)  # Light blue text color
        imgui.text("< Keyboard Controls >")
        imgui.pop_style_color()
        imgui.separator()

        # Display controls with highlighted keys
        imgui.bullet_text("[ESC or Q]: Quit the viewer")
        imgui.bullet_text("[W]: Toggle polygon mode (line, point, fill)")
        imgui.bullet_text("[R]: Toggle rotation")
        imgui.bullet_text("[D]: Toggle depth view")
        imgui.bullet_text("[S]: Save current frame")
        imgui.bullet_text("[-->]: Switch to next camera")  # Right Arrow
        imgui.bullet_text("[<--]: Switch to previous camera")  # Left Arrow
        imgui.bullet_text("[X]: Reset to default camera")

        # Add a button with conditional logic
        if self.depth:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.8, 0.2, 1.0)  # Green button
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.1, 0.6, 0.1, 1.0)  # Darker green hover
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.1, 0.5, 0.1, 1.0)  # Even darker green active

            if imgui.button("Save Frame"):
                self.save = True

            imgui.pop_style_color(3)

        # imgui.separator()
        
        # imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2, 1.0)  # Red button
        # imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.6, 0.1, 0.1, 1.0)  # Darker red hover
        # imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.5, 0.1, 0.1, 1.0)  # Even darker red active
        # if imgui.button("Next object"):
        #         tmp = self.drawables.pop(-1)
        #         self.drawables.insert(self.nums_cam, tmp)
        #         self.depth_drawables.insert(0, self.depth_drawables.pop(-1))
        # imgui.pop_style_color(3)
        
        imgui.separator()

        # Display current camera information with a highlight
        if self.cam_index != -1:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.7, 0.2, 1.0)  # Golden text for camera info
            if self.cam_index == 0:
                imgui.text("Current Camera: Master Camera")
            else:
                imgui.text(f"Current Camera: {self.cam_index}")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.8, 0.2, 0.2, 1.0)  # Red text for no camera
            imgui.text("Current Camera: None")
            imgui.pop_style_color()

        imgui.end()

    
    def add(self, depth, *drawables):
        """ add objects to draw in this windows """
        if depth:
            self.depth_drawables.extend(drawables)
            return
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """Handle keyboard events for controlling the viewer"""
        nums_keys = [glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5, glfw.KEY_6, glfw.KEY_7, glfw.KEY_8, glfw.KEY_9]
        if action in {glfw.PRESS, glfw.REPEAT}:
            if key in {glfw.KEY_ESCAPE, glfw.KEY_Q}:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_R:
                self.rotate = not self.rotate
            if key == glfw.KEY_D:
                self.depth = not self.depth
            if key == glfw.KEY_S:
                self.save = True
            if key == glfw.KEY_RIGHT:
                self.cam_index = (self.cam_index + 1) % len(self.cams)
            if key == glfw.KEY_LEFT:
                self.cam_index = (self.cam_index - 1) % len(self.cams)
            if key == glfw.KEY_X:
                self.cam_index = -1
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
        
    def on_resize(self, win, width, height):
        print('win', win, 'width', width, 'height', height)
        print("resize")
        self.win_size = (width, height)
        """Handle window resize events"""
        GL.glViewport(0, 0, width, height)
        
    def on_idle(self, _win):
        """Handle idle callback for updating the viewer"""
        print("idle")
        pass


# -------------- main program and scene setup --------------------------------
def main():
    """ create windows, add shaders & scene objects, then run rendering loop """
    viewer = Viewer()
    viewer.add(True, Cat("triangle\depth.vert", "triangle\depth.frag").setup())
    viewer.add(False, Cat("triangle/cat.vert", "triangle/cat.frag").setup())
    # viewer.add(True, Car("triangle\depth.vert", "triangle\depth.frag").setup())
    # viewer.add(False, Car("triangle/cat.vert", "triangle/cat.frag").setup())
    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize windows system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
