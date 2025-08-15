import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import ctypes

# Vertex and fragment shaders
vertex_shader = """
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

fragment_shader = """
#version 330 core
out vec4 FragColor;
uniform vec3 color;

void main()
{
    FragColor = vec4(color, 1.0);
}
"""

def compile_shaders():
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return shader

def create_cube():
    vertices = [
        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5,
        -0.5, -0.5,  0.5,
         0.5, -0.5,  0.5,
         0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5
    ]
    indices = [
        0, 1, 2, 2, 3, 0,
        1, 5, 6, 6, 2, 1,
        5, 4, 7, 7, 6, 5,
        4, 0, 3, 3, 7, 4,
        3, 2, 6, 6, 7, 3,
        4, 5, 1, 1, 0, 4
    ]
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def setup_buffers(vertices, indices):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)
    return vao, len(indices)

def draw_cube(shader, vao, count, model, scale, color):
    glUniform3fv(glGetUniformLocation(shader, "color"), 1, glm.value_ptr(color))
    model = glm.scale(model, scale)
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)

def draw_walls(shader, vao, count, scene_rotation):
    wall_color = glm.vec3(0.8, 0.8, 0.8)  # Light gray classroom walls

    # Back wall (behind blackboard)
    draw_cube(shader, vao, count,
              scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, 3, -5)),
              glm.vec3(15, 7, 0.1), wall_color)

    # Front wall (behind camera)
    # draw_cube(shader, vao, count,
    #           scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, 2, 10)),
    #           glm.vec3(10, 3, 0.1), wall_color)

    # Left wall
    draw_cube(shader, vao, count,
              scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(-7.5, 3, 2.5)),
              glm.vec3(0.1, 7, 15), wall_color)

    # Right wall
    draw_cube(shader, vao, count,
              scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(7.5, 3, 2.5)),
              glm.vec3(0.1, 7, 15), wall_color)


def draw_table(shader, vao, count, base_pos, scene_rotation):
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(0, 0.5, 0)), glm.vec3(2, 0.1, 1), glm.vec3(0.5, 0.3, 0.2))
    for dx in [-0.9, 0.9]:
        for dz in [-0.4, 0.4]:
            draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(dx, 0.25, dz)), glm.vec3(0.1, 0.5, 0.1), glm.vec3(0.3, 0.2, 0.1))

def draw_chair(shader, vao, count, base_pos, scene_rotation):
    # Seat
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(0, 0.3, 0)), glm.vec3(0.6, 0.1, 0.6), glm.vec3(0.2, 0.2, 0.6))
    # Back support frame
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(0, 0.65, -0.25)), glm.vec3(0.6, 0.5, 0.1), glm.vec3(0.2, 0.2, 0.6))
    # Arm rests
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(-0.3, 0.45, 0)), glm.vec3(0.1, 0.2, 0.6), glm.vec3(0.2, 0.2, 0.6))
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(0.3, 0.45, 0)), glm.vec3(0.1, 0.2, 0.6), glm.vec3(0.2, 0.2, 0.6))
    # Chair legs
    for dx in [-0.25, 0.25]:
        for dz in [-0.25, 0.25]:
            draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(-1.0), base_pos + glm.vec3(dx, 0.15, dz)), glm.vec3(0.1, 0.3, 0.1), glm.vec3(0.1, 0.1, 0.3))

def draw_blackboard(shader, vao, count, scene_rotation):
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, 2, -4.9)), glm.vec3(5, 3, 0.1), glm.vec3(0.0, 0.0, 0.0))

# Floor
def draw_floor(shader, vao, count, scene_rotation):
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, -0.1, 3)), glm.vec3(15, 0.1, 14), glm.vec3(0.3, 0.3, 0.3))

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "3D Classroom", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    shader = compile_shaders()
    vertices, indices = create_cube()
    vao, count = setup_buffers(vertices, indices)

    glUseProgram(shader)

    projection = glm.perspective(glm.radians(45), 800 / 600, 0.1, 100)
    view = glm.lookAt(glm.vec3(0, 4, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))

    rotation_angle = 0.0
    camera_pos = glm.vec3(0, 4, 10)

    def key_callback(window, key, scancode, action, mods):
        nonlocal rotation_angle, camera_pos
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_LEFT:
                rotation_angle -= 2.0
            elif key == glfw.KEY_RIGHT:
                rotation_angle += 2.0
            elif key == glfw.KEY_W:
                camera_pos.z -= 0.5
            elif key == glfw.KEY_S:
                camera_pos.z += 0.5
            elif key == glfw.KEY_A:
                camera_pos.x -= 0.5
            elif key == glfw.KEY_D:
                camera_pos.x += 0.5

    glfw.set_key_callback(window, key_callback)

    while not glfw.window_should_close(window):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = glm.lookAt(camera_pos, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))

        scene_rotation = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angle), glm.vec3(0, 1, 0))

        draw_floor(shader, vao, count, scene_rotation)
        draw_walls(shader, vao, count, scene_rotation)
        draw_blackboard(shader, vao, count, scene_rotation)
        draw_table(shader, vao, count, glm.vec3(0, 0, -2), scene_rotation)
        # model = glm.rotate(model, glm.radians(180), glm.vec3(0, 1, 0))

        for row in range(3):
            for col in range(3):
                x = -3 + col * 3
                z = 2 + row * 2+1
                draw_chair(shader, vao, count, glm.vec3(x, 0, z), scene_rotation)
                draw_table(shader, vao, count, glm.vec3(x, 0, z-1), scene_rotation)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()