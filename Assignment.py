# classroom_lit.py
# Run: python classroom_lit.py
# Deps: pip install glfw PyOpenGL PyGLM numpy

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glm
import numpy as np
from math import sin, cos, radians

# =========================
# Shaders (Phong lighting)
# =========================

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(model))) * aNormal; // normal matrix
    gl_Position = projection * view * worldPos;
}
"""

FRAGMENT_SHADER = """
#version 330 core

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool enabled;
};

struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
    bool enabled;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool enabled;
};

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 viewPos;
uniform Material material;
uniform DirLight dirLight;
uniform PointLight pointLight;
uniform SpotLight spotLight;

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir){
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);

    vec3 ambient  = light.ambient  * material.ambient;
    vec3 diffuse  = light.diffuse  * diff * material.diffuse;
    vec3 specular = light.specular * spec * material.specular;
    return ambient + diffuse + specular;
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 viewDir){
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);

    float distance = length(light.position - FragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance +
                               light.quadratic * (distance * distance));

    vec3 ambient  = light.ambient  * material.ambient;
    vec3 diffuse  = light.diffuse  * diff * material.diffuse;
    vec3 specular = light.specular * spec * material.specular;

    return (ambient + diffuse + specular) * attenuation;
}

vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 viewDir){
    vec3 lightDir = normalize(light.position - FragPos);
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

    float diff = max(dot(normal, lightDir), 0.0);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);

    vec3 ambient  = light.ambient  * material.ambient;
    vec3 diffuse  = light.diffuse  * diff * material.diffuse;
    vec3 specular = light.specular * spec * material.specular;

    return (ambient + diffuse + specular) * intensity;
}

void main(){
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 result = vec3(0.0);

    if (dirLight.enabled)
        result += CalcDirLight(dirLight, norm, viewDir);
    if (pointLight.enabled)
        result += CalcPointLight(pointLight, norm, viewDir);
    if (spotLight.enabled)
        result += CalcSpotLight(spotLight, norm, viewDir);

    FragColor = vec4(result, 1.0);
}
"""


# =========================
# Camera
# =========================

class Camera:
    def __init__(self, position=glm.vec3(0.0, 3.0, 10.0)):
        self.position = glm.vec3(position)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 7.0
        self.sensitivity = 0.1
        self.first_mouse = True
        self.lastX = 400
        self.lastY = 300

    def get_view(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def process_keyboard(self, window, dt):
        velocity = self.speed * dt
        right = glm.normalize(glm.cross(self.front, self.up))
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position += self.front * velocity
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position -= self.front * velocity
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.position -= right * velocity
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.position += right * velocity
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            self.position += self.up * velocity
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.position -= self.up * velocity

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.lastX = xpos
            self.lastY = ypos
            self.first_mouse = False
        xoffset = (xpos - self.lastX) * self.sensitivity
        yoffset = (self.lastY - ypos) * self.sensitivity
        self.lastX = xpos
        self.lastY = ypos

        self.yaw += xoffset
        self.pitch += yoffset
        self.pitch = max(-89.0, min(89.0, self.pitch))

        front = glm.vec3()
        front.x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.yaw)) * cos(radians(self.pitch))
        self.front = glm.normalize(front)

# =========================
# Geometry: Cube with normals
# =========================

def cube_vertices_with_normals():
    # 36 vertices (6 faces * 2 triangles * 3 verts), each: position (3) + normal (3)
    data = np.array([
        # back face (0,0,-1)
        -0.5,-0.5,-0.5,  0,0,-1,   0.5,-0.5,-0.5,  0,0,-1,   0.5, 0.5,-0.5,  0,0,-1,
         0.5, 0.5,-0.5,  0,0,-1,  -0.5, 0.5,-0.5,  0,0,-1,  -0.5,-0.5,-0.5,  0,0,-1,

        # front face (0,0,1)
        -0.5,-0.5, 0.5,  0,0, 1,   0.5,-0.5, 0.5,  0,0, 1,   0.5, 0.5, 0.5,  0,0, 1,
         0.5, 0.5, 0.5,  0,0, 1,  -0.5, 0.5, 0.5,  0,0, 1,  -0.5,-0.5, 0.5,  0,0, 1,

        # left face (-1,0,0)
        -0.5, 0.5, 0.5, -1,0,0,  -0.5, 0.5,-0.5, -1,0,0,  -0.5,-0.5,-0.5, -1,0,0,
        -0.5,-0.5,-0.5, -1,0,0,  -0.5,-0.5, 0.5, -1,0,0,  -0.5, 0.5, 0.5, -1,0,0,

        # right face (1,0,0)
         0.5, 0.5, 0.5,  1,0,0,   0.5, 0.5,-0.5,  1,0,0,   0.5,-0.5,-0.5,  1,0,0,
         0.5,-0.5,-0.5,  1,0,0,   0.5,-0.5, 0.5,  1,0,0,   0.5, 0.5, 0.5,  1,0,0,

        # bottom face (0,-1,0)
        -0.5,-0.5,-0.5,  0,-1,0,   0.5,-0.5,-0.5,  0,-1,0,   0.5,-0.5, 0.5,  0,-1,0,
         0.5,-0.5, 0.5,  0,-1,0,  -0.5,-0.5, 0.5,  0,-1,0,  -0.5,-0.5,-0.5,  0,-1,0,

        # top face (0,1,0)
        -0.5, 0.5,-0.5,  0, 1,0,   0.5, 0.5,-0.5,  0, 1,0,   0.5, 0.5, 0.5,  0, 1,0,
         0.5, 0.5, 0.5,  0, 1,0,  -0.5, 0.5, 0.5,  0, 1,0,  -0.5, 0.5,-0.5,  0, 1,0,
    ], dtype=np.float32)
    return data

def make_shader():
    return compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

# =========================
# Draw helpers
# =========================

def model_matrix_euler(pos=glm.vec3(0), rx=0.0, ry=0.0, rz=0.0, scale=glm.vec3(1.0)):
    m = glm.mat4(1.0)
    m = glm.translate(m, pos)
    if rx != 0.0:
        m = glm.rotate(m, glm.radians(rx), glm.vec3(1,0,0))
    if ry != 0.0:
        m = glm.rotate(m, glm.radians(ry), glm.vec3(0,1,0))
    if rz != 0.0:
        m = glm.rotate(m, glm.radians(rz), glm.vec3(0,0,1))
    m = glm.scale(m, scale)
    return m


def set_material(shader, ambient, diffuse, specular, shininess=32.0):
    glUniform3fv(glGetUniformLocation(shader, "material.ambient"), 1, glm.value_ptr(ambient))
    glUniform3fv(glGetUniformLocation(shader, "material.diffuse"), 1, glm.value_ptr(diffuse))
    glUniform3fv(glGetUniformLocation(shader, "material.specular"),1, glm.value_ptr(specular))
    glUniform1f(glGetUniformLocation(shader, "material.shininess"), shininess)

def draw_cube(shader, vao, model, color):
    set_material(shader, color * 0.3, color, glm.vec3(0.7), 32.0)
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 36)

def model_matrix(pos=glm.vec3(0), rot_y_deg=0.0, scale=glm.vec3(1.0)):
    m = glm.mat4(1.0)
    m = glm.translate(m, pos)
    m = glm.rotate(m, glm.radians(rot_y_deg), glm.vec3(0,1,0))
    m = glm.scale(m, scale)
    return m

def draw_table(shader, vao, base_pos):
    # tabletop
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3(0,0.80,0), 0, glm.vec3(2.0, 0.10, 1.2)),
              glm.vec3(0.55, 0.35, 0.20))
    # 4 legs
    leg_dx = 0.9; leg_dz = 0.5
    for dx in (-leg_dx, leg_dx):
        for dz in (-leg_dz, leg_dz):
            p = base_pos + glm.vec3(dx, 0.40, dz)
            draw_cube(shader, vao, model_matrix(p, 0, glm.vec3(0.10, 0.80, 0.10)),
                      glm.vec3(0.35, 0.22, 0.12))

def draw_chair(shader, vao, base_pos):
    # Colors
    metal = glm.vec3(0.70, 0.72, 0.76)
    plastic = glm.vec3(0.18, 0.22, 0.55)   # body
    cushion = glm.vec3(0.12, 0.55, 0.65)   # seat pad
    wood = glm.vec3(0.32, 0.22, 0.12)      # arm tops / slats

    # --- seat base ---
    seat_size = glm.vec3(0.80, 0.08, 0.80)
    seat_y = 0.48
    draw_cube(shader, vao,
              model_matrix(base_pos + glm.vec3(0, seat_y, 0), 0, seat_size),
              plastic)

    # --- seat cushion (slightly larger, thinner) ---
    draw_cube(shader, vao,
              model_matrix(base_pos + glm.vec3(0, seat_y + 0.06, 0), 0, glm.vec3(0.84, 0.06, 0.84)),
              cushion)

    # --- backrest frame posts (rear left/right) ---
    post_h = 0.85
    post_size = glm.vec3(0.08, post_h, 0.08)
    back_z = -0.36
    draw_cube(shader, vao,
              model_matrix(base_pos + glm.vec3(-0.34, seat_y + post_h/2.0, back_z), 0, post_size),
              metal)
    draw_cube(shader, vao,
              model_matrix(base_pos + glm.vec3( 0.34, seat_y + post_h/2.0, back_z), 0, post_size),
              metal)

    # --- angled backrest slats (3 horizontal) ---
    back_tilt = -10.0  # degrees around X
    for i, yoff in enumerate([0.10, 0.32, 0.54]):
        draw_cube(shader, vao,
            model_matrix_euler(base_pos + glm.vec3(0, seat_y + yoff + 0.35, back_z - 0.01),
                               rx=back_tilt, scale=glm.vec3(0.72, 0.06, 0.18)),
            wood)

    # --- armrest posts (front) ---
    arm_h = 0.28
    arm_y = seat_y + 0.20
    post = glm.vec3(0.06, arm_h, 0.06)
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3(-0.34, arm_y,  0.30), 0, post), metal)
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3( 0.34, arm_y,  0.30), 0, post), metal)

    # --- armrest tops (slight back tilt) ---
    draw_cube(shader, vao,
              model_matrix_euler(base_pos + glm.vec3(-0.34, arm_y + arm_h/2.0 + 0.06, 0.10),
                                 rx=-5.0, scale=glm.vec3(0.10, 0.06, 0.45)),
              wood)
    draw_cube(shader, vao,
              model_matrix_euler(base_pos + glm.vec3( 0.34, arm_y + arm_h/2.0 + 0.06, 0.10),
                                 rx=-5.0, scale=glm.vec3(0.10, 0.06, 0.45)),
              wood)

    # --- legs: splayed slightly (metal) ---
    leg_len = 0.46
    leg_size = glm.vec3(0.07, leg_len, 0.07)
    splay = 8.0  # degrees
    # front left/right
    draw_cube(shader, vao,
        model_matrix_euler(base_pos + glm.vec3(-0.30, leg_len/2.0, 0.30),
                           rx= splay, rz= splay, scale=leg_size), metal)
    draw_cube(shader, vao,
        model_matrix_euler(base_pos + glm.vec3( 0.30, leg_len/2.0, 0.30),
                           rx= splay, rz=-splay, scale=leg_size), metal)
    # back left/right
    draw_cube(shader, vao,
        model_matrix_euler(base_pos + glm.vec3(-0.30, leg_len/2.0,-0.30),
                           rx=-splay, rz= splay, scale=leg_size), metal)
    draw_cube(shader, vao,
        model_matrix_euler(base_pos + glm.vec3( 0.30, leg_len/2.0,-0.30),
                           rx=-splay, rz=-splay, scale=leg_size), metal)

    # --- stabilizer bars between legs ---
    bar_y = 0.18
    # left-right bars (front & back)
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3(0, bar_y,  0.30), 0, glm.vec3(0.62, 0.05, 0.05)), metal)
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3(0, bar_y, -0.30), 0, glm.vec3(0.62, 0.05, 0.05)), metal)
    # front-back bars (left & right)
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3(-0.30, bar_y, 0.0), 0, glm.vec3(0.05, 0.05, 0.62)), metal)
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3( 0.30, bar_y, 0.0), 0, glm.vec3(0.05, 0.05, 0.62)), metal)

    # --- optional footrest bar (front) ---
    draw_cube(shader, vao, model_matrix(base_pos + glm.vec3(0, 0.12, 0.34), 0, glm.vec3(0.60, 0.05, 0.05)), metal)


# =========================
# Main
# =========================

def main():
    # Window
    if not glfw.init():
        raise SystemExit("Failed to init GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(1280, 720, "3D Classroom - Camera & Lights", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Failed to create window")
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    # GL state
    glEnable(GL_DEPTH_TEST)

    # Shader
    shader = make_shader()
    glUseProgram(shader)

    # Geometry buffers
    vertices = cube_vertices_with_normals()
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    stride = 6 * 4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    # Camera
    camera = Camera()

    # Mouse callback
    def mouse_cb(win, xpos, ypos):
        camera.process_mouse(xpos, ypos)
    glfw.set_cursor_pos_callback(window, mouse_cb)

    # Projection
    projection = glm.perspective(glm.radians(45.0), 1280/720, 0.1, 200.0)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

    # Light toggles
    dir_enabled = True
    point_enabled = True
    spot_enabled = True  # flashlight
    flashlight_toggle_debounce = False

    # Time
    last_time = glfw.get_time()

    print("""
=== 3D Classroom ===
Controls:
  Mouse        -> look
  W/A/S/D      -> move
  SPACE / LSHIFT -> up / down
  F            -> toggle flashlight (spot light)
  1 / 2 / 3    -> toggle directional / point / spot
  ESC          -> quit
""")

    while not glfw.window_should_close(window):
        # Timing
        t = glfw.get_time()
        dt = t - last_time
        last_time = t

        # Input
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        camera.process_keyboard(window, dt)

        # Toggle lights
        if glfw.get_key(window, glfw.KEY_F) == glfw.PRESS:
            if not flashlight_toggle_debounce:
                spot_enabled = not spot_enabled
                flashlight_toggle_debounce = True
        else:
            flashlight_toggle_debounce = False

        if glfw.get_key(window, glfw.KEY_1) == glfw.PRESS:
            dir_enabled = True
        if glfw.get_key(window, glfw.KEY_2) == glfw.PRESS:
            point_enabled = True
        if glfw.get_key(window, glfw.KEY_3) == glfw.PRESS:
            spot_enabled = True

        if glfw.get_key(window, glfw.KEY_1) == glfw.RELEASE and glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            dir_enabled = False
        if glfw.get_key(window, glfw.KEY_2) == glfw.RELEASE and glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            point_enabled = False
        if glfw.get_key(window, glfw.KEY_3) == glfw.RELEASE and glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            spot_enabled = False

        # Clear
        glViewport(0, 0, 1280, 720)
        glClearColor(0.08, 0.09, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # View & view position
        view = camera.get_view()
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, glm.value_ptr(camera.position))

        # Lights
        # Directional (like sun from upper-left front)
        glUniform3f(glGetUniformLocation(shader, "dirLight.direction"), -0.3, -1.0, -0.2)
        glUniform3f(glGetUniformLocation(shader, "dirLight.ambient"),  0.20, 0.20, 0.22)
        glUniform3f(glGetUniformLocation(shader, "dirLight.diffuse"),  0.55, 0.55, 0.55)
        glUniform3f(glGetUniformLocation(shader, "dirLight.specular"), 0.80, 0.80, 0.80)
        glUniform1i(glGetUniformLocation(shader, "dirLight.enabled"), 1 if dir_enabled else 0)

        # Point (ceiling bulb in center)
        glUniform3f(glGetUniformLocation(shader, "pointLight.position"), 0.0, 3.8, 0.0)
        glUniform3f(glGetUniformLocation(shader, "pointLight.ambient"),  0.05, 0.05, 0.05)
        glUniform3f(glGetUniformLocation(shader, "pointLight.diffuse"),  1.00, 0.95, 0.85)
        glUniform3f(glGetUniformLocation(shader, "pointLight.specular"), 1.00, 1.00, 1.00)
        glUniform1f(glGetUniformLocation(shader, "pointLight.constant"),  1.0)
        glUniform1f(glGetUniformLocation(shader, "pointLight.linear"),    0.09)
        glUniform1f(glGetUniformLocation(shader, "pointLight.quadratic"), 0.032)
        glUniform1i(glGetUniformLocation(shader, "pointLight.enabled"), 1 if point_enabled else 0)

        # Spotlight (flashlight from camera)
        glUniform3fv(glGetUniformLocation(shader, "spotLight.position"), 1, glm.value_ptr(camera.position))
        glUniform3fv(glGetUniformLocation(shader, "spotLight.direction"),1, glm.value_ptr(camera.front))
        glUniform1f(glGetUniformLocation(shader, "spotLight.cutOff"),  glm.cos(glm.radians(12.5)))
        glUniform1f(glGetUniformLocation(shader, "spotLight.outerCutOff"), glm.cos(glm.radians(18.0)))
        glUniform3f(glGetUniformLocation(shader, "spotLight.ambient"),  0.00, 0.00, 0.00)
        glUniform3f(glGetUniformLocation(shader, "spotLight.diffuse"),  1.00, 1.00, 1.00)
        glUniform3f(glGetUniformLocation(shader, "spotLight.specular"), 1.00, 1.00, 1.00)
        glUniform1i(glGetUniformLocation(shader, "spotLight.enabled"), 1 if spot_enabled else 0)

        # Draw scene
        glBindVertexArray(VAO)

        # Floor
        draw_cube(shader, VAO, model_matrix(glm.vec3(0,-0.05,0), 0, glm.vec3(12, 0.1, 12)),
                  glm.vec3(0.50,0.50,0.52))

        # Ceiling
        draw_cube(shader, VAO, model_matrix(glm.vec3(0,4.05,0), 0, glm.vec3(12, 0.1, 12)),
                  glm.vec3(0.85,0.85,0.88))

        # Back wall
        draw_cube(shader, VAO, model_matrix(glm.vec3(0,2,-6.0), 0, glm.vec3(12, 4, 0.1)),
                  glm.vec3(0.75,0.78,0.90))
        # Left wall
        draw_cube(shader, VAO, model_matrix(glm.vec3(-6.0,2,0), 0, glm.vec3(0.1, 4, 12)),
                  glm.vec3(0.76,0.80,0.92))
        # Right wall
        draw_cube(shader, VAO, model_matrix(glm.vec3( 6.0,2,0), 0, glm.vec3(0.1, 4, 12)),
                  glm.vec3(0.76,0.80,0.92))

        # Blackboard on back wall
        draw_cube(shader, VAO, model_matrix(glm.vec3(0,2.3,-5.95), 0, glm.vec3(4.5, 1.6, 0.05)),
                  glm.vec3(0.02,0.18,0.02))
        # Blackboard frame
        draw_cube(shader, VAO, model_matrix(glm.vec3(0,2.3,-5.99), 0, glm.vec3(4.8, 1.8, 0.02)),
                  glm.vec3(0.25,0.15,0.06))

        # Teacher desk (front center)
        draw_table(shader, VAO, glm.vec3(0,0,-2.2))

        # Student rows (3x3)
        start_x = -3.5
        start_z =  1.5
        dx = 3.5
        dz = 2.5
        for r in range(3):
            for c in range(3):
                px = start_x + c*dx
                pz = start_z + r*dz
                # chair
                draw_chair(shader, VAO, glm.vec3(px, 0, pz))
                # table in front of chair
                draw_table(shader, VAO, glm.vec3(px, 0, pz - 1.2))

        # A simple “bulb” mesh showing point light position
        draw_cube(shader, VAO, model_matrix(glm.vec3(0,3.8,0), 0, glm.vec3(0.2,0.2,0.2)),
                  glm.vec3(1.0, 0.95, 0.8))

        # Done
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup
    glDeleteBuffers(1, [VBO])
    glDeleteVertexArrays(1, [VAO])
    glDeleteProgram(shader)
    glfw.terminate()

if __name__ == "__main__":
    main()
