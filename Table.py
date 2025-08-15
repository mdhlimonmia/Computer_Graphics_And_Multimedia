import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm

# === Vertex Shader ===
# Transforms each vertex by the model, view, and projection matrices
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# === Fragment Shader ===
# Sets the output color for each pixel
fragment_shader = """
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(0.7, 0.4, 0.1, 1.0); // Brownish table color
}
"""

# Compiles and links shaders into a program
def create_shader():
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return shader

# Defines the vertices and indices for a cube
def create_cube():
    vertices = [
        -0.5, -0.5, -0.5,  # 0
         0.5, -0.5, -0.5,  # 1
         0.5,  0.5, -0.5,  # 2
        -0.5,  0.5, -0.5,  # 3
        -0.5, -0.5,  0.5,  # 4
         0.5, -0.5,  0.5,  # 5
         0.5,  0.5,  0.5,  # 6
        -0.5,  0.5,  0.5   # 7
    ]
    indices = [
        0, 1, 2, 2, 3, 0,  # back face
        1, 5, 6, 6, 2, 1,  # right face
        5, 4, 7, 7, 6, 5,  # front face
        4, 0, 3, 3, 7, 4,  # left face
        3, 2, 6, 6, 7, 3,  # top face
        0, 4, 5, 5, 1, 0   # bottom face
    ]
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

# Sets up VAO/VBO/EBO for rendering the cube
def setup_cube_vao(vertices, indices):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    # Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Define vertex attributes (position only)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)
    return vao, len(indices)

def main():
    # === Initialize GLFW ===
    if not glfw.init():
        return

    # Set OpenGL version to 3.3 core profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create a window
    window = glfw.create_window(800, 600, "OpenGL Table", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)  # Enable depth for 3D rendering

    # === Setup shaders and geometry ===
    shader = create_shader()
    vertices, indices = create_cube()
    cube_vao, index_count = setup_cube_vao(vertices, indices)

    glUseProgram(shader)

    # === Set up camera ===
    projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
    view = glm.lookAt(glm.vec3(2, 2, 5), glm.vec3(0, 0.5, 0), glm.vec3(0, 1, 0))

    # Send matrices to shader
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))

    # === Key press to close window ===
    def key_callback(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    glfw.set_key_callback(window, key_callback)

    # === Render loop ===
    while not glfw.window_should_close(window):
        glClearColor(0.1, 0.1, 0.15, 1.0)  # Background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)
        glBindVertexArray(cube_vao)

        # === Draw the tabletop ===
        model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 1.0, 0.0))  # Move up
        model = glm.scale(model, glm.vec3(2.0, 0.1, 1.2))              # Make wide and flat
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

        # === Draw the 4 legs ===
        leg_positions = [
            (-0.9, 0.5, -0.5),  # Front-left
            ( 0.9, 0.5, -0.5),  # Front-right
            (-0.9, 0.5,  0.5),  # Back-left
            ( 0.9, 0.5,  0.5)   # Back-right
        ]
        for pos in leg_positions:
            model = glm.translate(glm.mat4(1.0), glm.vec3(*pos))       # Move to leg position
            model = glm.scale(model, glm.vec3(0.1, 1.0, 0.1))          # Thin and tall
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
            glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

        # === Finish frame ===
        glfw.swap_buffers(window)
        glfw.poll_events()

    # === Cleanup ===
    glDeleteVertexArrays(1, [cube_vao])
    glDeleteProgram(shader)
    glfw.terminate()

if __name__ == "__main__":
    main()
