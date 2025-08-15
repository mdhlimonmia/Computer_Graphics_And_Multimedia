import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm

print("Imported glm from:", glm.__file__)

# Vertex Shader (transforms vertex positions using model-view-projection matrix)
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# Fragment Shader (outputs a solid red color for all fragments)
fragment_shader_source = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color
}
"""

# Compile and link shaders into a program
def create_shader_program():
    vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader_program = compileProgram(vertex_shader, fragment_shader)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program

# Define cube vertex positions and indices for indexed drawing
def create_cube():
    vertices = [
        -0.5, -0.5, -0.5, # 0
         0.5, -0.5, -0.5, # 1
         0.5,  0.5, -0.5, # 2
        -0.5,  0.5, -0.5, # 3
        -0.5, -0.5,  0.5, # 4
         0.5, -0.5,  0.5, # 5
         0.5,  0.5,  0.5, # 6
        -0.5,  0.5,  0.5  # 7
    ]
    indices = [
        0, 1, 2, 2, 3, 0,       # Back face
        1, 5, 6, 6, 2, 1,       # Right face
        5, 4, 7, 7, 6, 5,       # Front face
        4, 0, 3, 3, 7, 4,       # Left face
        3, 2, 6, 6, 7, 3,       # Top face
        0, 4, 5, 5, 1, 0        # Bottom face
    ]
    return vertices, indices

# Upload cube data to GPU and configure VAO/VBO/EBO
def setup_vao(vertices, indices):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    # Upload vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

    # Upload index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(indices, dtype=np.uint32), GL_STATIC_DRAW)

    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)

    return vao, len(indices)

def main():
    if not glfw.init():
        return

    # Create a window with OpenGL context
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Base Cube", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)  # Enable depth testing

    # Compile shaders and create geometry
    shader_program = create_shader_program()
    cube_vertices, cube_indices = create_cube()
    cube_vao, cube_index_count = setup_vao(cube_vertices, cube_indices)
    
    glUseProgram(shader_program)

    # Set up projection and view matrices
    projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

    view = glm.lookAt(glm.vec3(0.0, 2.0, 5.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))

    # Handle key press events
    def key_callback(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, key_callback)

    angle = 0.0

    # Render loop
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        # 1. Base cube at origin
        model = glm.mat4(1.0)
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glBindVertexArray(cube_vao)
        glDrawElements(GL_TRIANGLES, cube_index_count, GL_UNSIGNED_INT, None)

        # # 2. Cube placed above the base cube
        # model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.5, 0.0))
        # glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        # glBindVertexArray(cube_vao)
        # glDrawElements(GL_TRIANGLES, cube_index_count, GL_UNSIGNED_INT, None)

        # # 3. Rotating cube on top
        # model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.5, 0.0))
        # model = glm.rotate(model, glm.radians(angle), glm.vec3(0.0, 1.0, 0.0))
        # glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        # glBindVertexArray(cube_vao)
        # glDrawElements(GL_TRIANGLES, cube_index_count, GL_UNSIGNED_INT, None)

        # # 4. Scaled cube (larger)
        # model = glm.translate(glm.mat4(1.0), glm.vec3(-0.5, 0.5, 0.0))
        # model = glm.scale(model, glm.vec3(1.5, 1.5, 1.5))
        # glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        # glBindVertexArray(cube_vao)
        # glDrawElements(GL_TRIANGLES, cube_index_count, GL_UNSIGNED_INT, None)

        # 5. Flat platform-like cube
        model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 0.0))
        model = glm.scale(model, glm.vec3(2.0, 0.1, 1.0))
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glBindVertexArray(cube_vao)
        glDrawElements(GL_TRIANGLES, cube_index_count, GL_UNSIGNED_INT, None)

        # 6. Table legs (scaled down vertical cubes at corners)
        leg_positions = [
            (-0.8, -0.5, -0.3),
            (0.8, -0.5, -0.3),
            (-0.8, -0.4, 0.3),
            (0.8, -0.4, 0.3)
        ]
        for pos in leg_positions:
            model = glm.translate(glm.mat4(1.0), glm.vec3(*pos))
            model = glm.scale(model, glm.vec3(0.1, 0.5, 0.1))
            glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
            glBindVertexArray(cube_vao)
            glDrawElements(GL_TRIANGLES, cube_index_count, GL_UNSIGNED_INT, None)

        # Increment angle for rotation animation
        angle += 1.0

        # Swap buffers and poll window events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup
    glDeleteVertexArrays(1, [cube_vao])
    glDeleteProgram(shader_program)
    glfw.terminate()

if __name__ == "__main__":
    main()
