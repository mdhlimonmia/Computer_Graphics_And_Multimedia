import glfw
from OpenGL.GL import *

def draw_triangle():
    #"""Draw a triangle with colored vertices"""
    glBegin(GL_TRIANGLES)
    # Top vertex - Red
    glColor3f(1.0, 0.0, 0.0) # Red
    glVertex2f(0.0, 0.5) # Top center
    # Bottom-left vertex - Green
    glColor3f(0.0, 1.0, 0.0) # Green
    glVertex2f(-0.5, -0.5) # Bottom-left

    # Bottom-right vertex - Blue
    glColor3f(0.0, 0.0, 1.0) # Blue
    glVertex2f(0.5, -0.5) # Bottom-right
    glEnd()

def main():
    # Initialize GLFW
    if not glfw.init():
        return
    # Create a window
    window = glfw.create_window(800, 600, "PyOpenGL Lab", None, None)
    if not window:
        glfw.terminate()
        return
    # Set the window as the current context
    glfw.make_context_current(window)
    # Set the clear color (black)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    # Rendering loop
    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        #Draw Triangle
        draw_triangle()

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

        # Clean up
    glfw.terminate()
if __name__ == "__main__":
    main()