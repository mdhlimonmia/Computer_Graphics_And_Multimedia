from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math

# Camera state
cam_pos = [0.0, 1.5, 5.0]
cam_yaw = 0.0
cam_pitch = 0.0

# Fan rotation angle
angle = 0.0

# Room size
ROOM_SIZE = 10.0

def draw_fan_blade():
    glPushMatrix()
    glScalef(0.1, 1.0, 0.02)
    glutSolidCube(1.0)
    glPopMatrix()

def draw_fan_hub():
    quad = gluNewQuadric()
    glColor3f(0.5, 0.5, 0.5)
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(quad, 0.2, 0.2, 0.2, 32, 32)
    glPopMatrix()

def draw_classroom():
    # Floor
    glColor3f(0.8, 0.8, 0.8)
    glBegin(GL_QUADS)
    glVertex3f(-ROOM_SIZE, 0, -ROOM_SIZE)
    glVertex3f( ROOM_SIZE, 0, -ROOM_SIZE)
    glVertex3f( ROOM_SIZE, 0,  ROOM_SIZE)
    glVertex3f(-ROOM_SIZE, 0,  ROOM_SIZE)
    glEnd()

    # Ceiling
    glColor3f(0.95, 0.95, 0.95)
    glBegin(GL_QUADS)
    glVertex3f(-ROOM_SIZE, 3, -ROOM_SIZE)
    glVertex3f( ROOM_SIZE, 3, -ROOM_SIZE)
    glVertex3f( ROOM_SIZE, 3,  ROOM_SIZE)
    glVertex3f(-ROOM_SIZE, 3,  ROOM_SIZE)
    glEnd()

    # Walls
    glColor3f(0.9, 0.9, 1.0)
    def wall(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
        glBegin(GL_QUADS)
        glVertex3f(x1, y1, z1)
        glVertex3f(x2, y2, z2)
        glVertex3f(x3, y3, z3)
        glVertex3f(x4, y4, z4)
        glEnd()

    # Back
    wall(-ROOM_SIZE, 0, -ROOM_SIZE,  ROOM_SIZE, 0, -ROOM_SIZE,
         ROOM_SIZE, 3, -ROOM_SIZE, -ROOM_SIZE, 3, -ROOM_SIZE)

    # Front
    wall(-ROOM_SIZE, 0, ROOM_SIZE,  ROOM_SIZE, 0, ROOM_SIZE,
         ROOM_SIZE, 3, ROOM_SIZE, -ROOM_SIZE, 3, ROOM_SIZE)

    # Left
    wall(-ROOM_SIZE, 0, -ROOM_SIZE, -ROOM_SIZE, 0, ROOM_SIZE,
         -ROOM_SIZE, 3, ROOM_SIZE, -ROOM_SIZE, 3, -ROOM_SIZE)

    # Right
    wall(ROOM_SIZE, 0, -ROOM_SIZE, ROOM_SIZE, 0, ROOM_SIZE,
         ROOM_SIZE, 3, ROOM_SIZE, ROOM_SIZE, 3, -ROOM_SIZE)

def display():
    global angle, cam_pos, cam_yaw, cam_pitch

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Camera direction
    dir_x = math.cos(math.radians(cam_pitch)) * math.sin(math.radians(cam_yaw))
    dir_y = math.sin(math.radians(cam_pitch))
    dir_z = -math.cos(math.radians(cam_pitch)) * math.cos(math.radians(cam_yaw))

    gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2],
              cam_pos[0] + dir_x, cam_pos[1] + dir_y, cam_pos[2] + dir_z,
              0.0, 1.0, 0.0)

    draw_classroom()

    # Fan
    glPushMatrix()
    glTranslatef(0.0, 2.95, 0.0)
    draw_fan_hub()
    for i in range(4):
        glPushMatrix()
        glRotatef(angle + i * 90, 0.0, 0.0, 1.0)
        glTranslatef(0.0, 0.6, 0.0)
        glColor3f(0.2, 0.2, 0.2)
        draw_fan_blade()
        glPopMatrix()
    glPopMatrix()

    glutSwapBuffers()

def idle():
    global angle
    angle += 1.0
    if angle > 360:
        angle -= 360
    glutPostRedisplay()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, float(w)/float(h), 1.0, 100.0)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    global cam_pos, cam_yaw, cam_pitch
    try:
        key = key.decode('utf-8')
    except:
        pass

    step = 0.2

    if key == 'w':
        cam_pos[0] += math.sin(math.radians(cam_yaw)) * step
        cam_pos[2] -= math.cos(math.radians(cam_yaw)) * step
    elif key == 's':
        cam_pos[0] -= math.sin(math.radians(cam_yaw)) * step
        cam_pos[2] += math.cos(math.radians(cam_yaw)) * step
    elif key == 'a':
        cam_yaw -= 5.0
    elif key == 'd':
        cam_yaw += 5.0
    elif key == 'q':
        cam_pitch += 5.0
        cam_pitch = min(cam_pitch, 89.0)
    elif key == 'e':
        cam_pitch -= 5.0
        cam_pitch = max(cam_pitch, -89.0)

def special_keys(key, x, y):
    global cam_pos
    step = 0.2
    if key == GLUT_KEY_LEFT:
        cam_pos[0] -= step  # strafe left
    elif key == GLUT_KEY_RIGHT:
        cam_pos[0] += step  # strafe right
    elif key == GLUT_KEY_UP:
        cam_pos[1] += step  # move up
    elif key == GLUT_KEY_DOWN:
        cam_pos[1] -= step  # move down

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1000, 700)
    glutCreateWindow(b"3D Classroom with Ceiling Fan")

    glEnable(GL_DEPTH_TEST)
    glClearColor(1.0, 1.0, 1.0, 1.0)

    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)

    glutMainLoop()

if __name__ == "__main__":
    main()
