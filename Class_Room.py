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
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  // Transform normal properly
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;        // Point light position
uniform vec3 spotLightPos;    // Spotlight position
uniform vec3 spotLightDir;    // Spotlight direction
uniform vec3 dirLightDir;     // Directional light direction

uniform float cutOff;
uniform float outerCutOff;

uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{
    // Ambient component (shared for simplicity)
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    // ====== Point Light ======
    vec3 lightDirPoint = normalize(lightPos - FragPos);
    float diffPoint = max(dot(norm, lightDirPoint), 0.0);
    vec3 reflectDirPoint = reflect(-lightDirPoint, norm);
    float specularStrength = 0.5;
    float specPoint = pow(max(dot(viewDir, reflectDirPoint), 0.0), 32);

    vec3 diffusePoint = diffPoint * lightColor;
    vec3 specularPoint = specularStrength * specPoint * lightColor;

    // ====== Spotlight ======
    vec3 lightDirSpot = normalize(spotLightPos - FragPos);
    float theta = dot(lightDirSpot, normalize(-spotLightDir));
    float intensity = 0.0;
    if(theta > outerCutOff) {
        if(theta > cutOff) {
            intensity = 1.0;
        } else {
            float epsilon = cutOff - outerCutOff;
            intensity = clamp((theta - outerCutOff) / epsilon, 0.0, 1.0);
        }
    }
    float diffSpot = max(dot(norm, lightDirSpot), 0.0);
    vec3 reflectDirSpot = reflect(-lightDirSpot, norm);
    float specSpot = pow(max(dot(viewDir, reflectDirSpot), 0.0), 32);

    vec3 diffuseSpot = diffSpot * lightColor * intensity;
    vec3 specularSpot = specularStrength * specSpot * lightColor * intensity;

    // ====== Directional Light ======
    vec3 lightDirDir = normalize(-dirLightDir); // directional light direction is the light ray direction
    float diffDir = max(dot(norm, lightDirDir), 0.0);
    vec3 reflectDirDir = reflect(-lightDirDir, norm);
    float specDir = pow(max(dot(viewDir, reflectDirDir), 0.0), 32);

    vec3 diffuseDir = diffDir * lightColor;
    vec3 specularDir = specularStrength * specDir * lightColor;

    // Combine all light contributions
    vec3 result = ambient * objectColor
                + (diffusePoint + diffuseSpot + diffuseDir) * objectColor
                + (specularPoint + specularSpot + specularDir);

    FragColor = vec4(result, 1.0);
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
        # positions          # normals
        -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,    0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,    0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,    0.0,  0.0, -1.0,

        -0.5, -0.5,  0.5,    0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,    0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,    0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,    0.0,  0.0,  1.0,

        -0.5,  0.5,  0.5,   -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5,   -1.0,  0.0,  0.0,

         0.5,  0.5,  0.5,    1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,    1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,    1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,    1.0,  0.0,  0.0,

        -0.5, -0.5, -0.5,    0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,    0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,    0.0, -1.0,  0.0,
        -0.5, -0.5,  0.5,    0.0, -1.0,  0.0,

        -0.5,  0.5, -0.5,    0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,    0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,    0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,    0.0,  1.0,  0.0
    ]

    indices = [
        0, 1, 2, 2, 3, 0,       # back face
        4, 5, 6, 6, 7, 4,       # front face
        8, 9, 10, 10, 11, 8,    # left face
        12, 13, 14, 14, 15, 12, # right face
        16, 17, 18, 18, 19, 16, # bottom face
        20, 21, 22, 22, 23, 20  # top face
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

    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)
    return vao, len(indices)

def draw_cube(shader, vao, count, model, scale, color):
    glUniform3fv(glGetUniformLocation(shader, "objectColor"), 1, glm.value_ptr(color))
    model = glm.scale(model, scale)
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)


def draw_walls(shader, vao, count, scene_rotation):
    wall_color = glm.vec3(0.8, 0.8, 0.8)  # Light gray classroom walls

    # Back wall (behind blackboard)
    draw_cube(shader, vao, count,
              scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, 3, -5)),
              glm.vec3(15, 5, 0.1), wall_color)

    # Front wall (behind camera)
    # draw_cube(shader, vao, count,
    #           scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, 2, 10)),
    #           glm.vec3(10, 3, 0.1), wall_color)

    # Left wall
    draw_cube(shader, vao, count,
              scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(-7.5, 3, 2.5)),
              glm.vec3(0.1, 5, 15), wall_color)

    # Right wall
    draw_cube(shader, vao, count,
              scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(7.5, 3, 2.5)),
              glm.vec3(0.1, 5, 15), wall_color)


def draw_table(shader, vao, count, base_pos, scene_rotation):
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(0, 0.5, 0)), glm.vec3(2, 0.1, 1), glm.vec3(0.5, 0.3, 0.2))
    for dx in [-0.9, 0.9]:
        for dz in [-0.4, 0.4]:
            draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), base_pos + glm.vec3(dx, 0.25, dz)), glm.vec3(0.1, 0.5, 0.1), glm.vec3(0.3, 0.2, 0.1))

def draw_chair(shader, vao, count, base_pos, scene_rotation, facing_dir):
    # Rotate chair 180 degrees around Y to flip direction
    flip_rotation = glm.rotate(glm.mat4(1.0), glm.radians(facing_dir), glm.vec3(0,1,0))
    transform = scene_rotation * glm.translate(glm.mat4(1.0), base_pos) * flip_rotation

    # Seat
    draw_cube(shader, vao, count, transform * glm.translate(glm.mat4(1.0), glm.vec3(0, 0.3, 0)), glm.vec3(0.6, 0.1, 0.6), glm.vec3(0.2, 0.2, 0.6))
    
    # Back support frame
    draw_cube(shader, vao, count, transform * glm.translate(glm.mat4(1.0), glm.vec3(0, 0.65, -0.25)), glm.vec3(0.6, 0.5, 0.1), glm.vec3(0.2, 0.2, 0.6))
    
    # Arm rests
    draw_cube(shader, vao, count, transform * glm.translate(glm.mat4(1.0), glm.vec3(-0.3, 0.45, 0)), glm.vec3(0.1, 0.2, 0.6), glm.vec3(0.2, 0.2, 0.6))
    draw_cube(shader, vao, count, transform * glm.translate(glm.mat4(1.0), glm.vec3(0.3, 0.45, 0)), glm.vec3(0.1, 0.2, 0.6), glm.vec3(0.2, 0.2, 0.6))
    
    # Legs
    leg_height = 0.3
    leg_scale = glm.vec3(0.1, leg_height, 0.1)
    leg_positions = [
        glm.vec3(-0.25, leg_height / 2, -0.25),
        glm.vec3(0.25, leg_height / 2, -0.25),
        glm.vec3(-0.25, leg_height / 2, 0.25),
        glm.vec3(0.25, leg_height / 2, 0.25),
    ]
    for pos in leg_positions:
        draw_cube(shader, vao, count, transform * glm.translate(glm.mat4(1.0), pos), leg_scale, glm.vec3(0.1, 0.1, 0.3))

def draw_blackboard(shader, vao, count, scene_rotation):
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, 3, -4)), glm.vec3(5, 3, 0.1), glm.vec3(0.0, 0.0, 0.0))

# Floor
def draw_floor(shader, vao, count, scene_rotation):
    draw_cube(shader, vao, count, scene_rotation * glm.translate(glm.mat4(1.0), glm.vec3(0, -0.1, 3)), glm.vec3(15, 0.1, 14), glm.vec3(0.3, 0.3, 0.3))

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    #Create window
    window = glfw.create_window(800, 600, "3D Classroom", None, None)
    if not window:
        glfw.terminate()
        print("There no an window")
        return
    
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    # shader Part
    shader = compile_shaders()
    vertices, indices = create_cube()
    vao, count = setup_buffers(vertices, indices)
    glUseProgram(shader)

    # Lighting
    light_pos_loc = glGetUniformLocation(shader, "lightPos")

    spot_light_pos_loc = glGetUniformLocation(shader, "spotLightPos")
    spot_light_dir_loc = glGetUniformLocation(shader, "spotLightDir")

    cut_off_loc = glGetUniformLocation(shader, "cutOff")
    outer_cut_off_loc = glGetUniformLocation(shader, "outerCutOff")
    view_pos_loc = glGetUniformLocation(shader, "viewPos")
    light_color_loc = glGetUniformLocation(shader, "lightColor")
    dir_light_dir_loc = glGetUniformLocation(shader, "dirLightDir")
    
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

    #Rendering Loop
    while not glfw.window_should_close(window):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Point light position
        point_light_pos = glm.vec3(5, 5, 5)
        glUniform3fv(light_pos_loc, 1, glm.value_ptr(point_light_pos))

        # Spotlight parameters
        spot_light_pos = glm.vec3(2, 5, 2)
        spot_light_dir = glm.normalize(glm.vec3(-0.5, -1, -0.5))
        glUniform3fv(spot_light_pos_loc, 1, glm.value_ptr(spot_light_pos))
        glUniform3fv(spot_light_dir_loc, 1, glm.value_ptr(spot_light_dir))

        cut_off = glm.cos(glm.radians(12.5))
        outer_cut_off = glm.cos(glm.radians(17.5))
        glUniform1f(cut_off_loc, cut_off)
        glUniform1f(outer_cut_off_loc, outer_cut_off)

        # Directional light direction
        dir_light_dir = glm.normalize(glm.vec3(-0.2, -1.0, -0.3))
        glUniform3fv(dir_light_dir_loc, 1, glm.value_ptr(dir_light_dir))

        # Camera position and light color
        glUniform3fv(view_pos_loc, 1, glm.value_ptr(camera_pos))
        glUniform3fv(light_color_loc, 1, glm.value_ptr(glm.vec3(1, 1, 1))) 



        view = glm.lookAt(camera_pos, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))

        scene_rotation = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angle), glm.vec3(0, 1, 0))

        draw_floor(shader, vao, count, scene_rotation)
        draw_walls(shader, vao, count, scene_rotation)
        draw_blackboard(shader, vao, count, scene_rotation)
        draw_chair(shader, vao, count, glm.vec3(0, 0, -2), scene_rotation, 0)
        draw_table(shader, vao, count, glm.vec3(0, 0, -1), scene_rotation)

        for row in range(3):
            for col in range(3):
                x = -3 + col * 3
                z = 2 + row * 2+1
                draw_chair(shader, vao, count, glm.vec3(x, 0, z), scene_rotation, 180)
                draw_table(shader, vao, count, glm.vec3(x, 0, z-1), scene_rotation)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
