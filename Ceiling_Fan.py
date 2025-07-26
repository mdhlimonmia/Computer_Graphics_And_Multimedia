from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math

# State variables
angle = 0.0       # Rotation angle
scale = 1.0       # Scaling factor
tx, ty = 0.0, 0.0 # Translation offsets

# Rotation Axis
x_ax = 0.0
y_ax = 0.0
z_ax = 1.0

#Rod Rotation Control
y_rod = 0.0
x_rod = 0.1
#Creade Blade
def draw_fan_blade():
    glBegin(GL_QUADS)
    glColor3f(0.4, 0.5, 0.7)  # Blade color
    glVertex2f(-0.1, 0.1)
    glVertex2f(0.2, 0.2)
    glVertex2f(0.1, 0.6)
    glVertex2f(-0.1, 0.6)
    glEnd()

#Create Rod
def draw_ceiling_rod():
    glColor3f(0.4, 0.4, 0.4)
    glBegin(GL_QUADS)
    glVertex2f(x_rod, y_rod)   # Top at ceiling (wider for visibility)
    glVertex2f(-x_rod, y_rod)
    glVertex2f( 0.00, 0.0)   # Bottom connects to fan hub
    glVertex2f(-0.00, 0.0)
    glEnd()

def display():
    glClear(GL_COLOR_BUFFER_BIT) #Clear old Color
    glLoadIdentity()

    #Apply translation and scaling to everything
    glTranslatef(tx, ty, 0.0)
    glScalef(scale, scale, 1.0)

    #Draw the rod fixed (no rotation)
    draw_ceiling_rod()

    # Now draw the fan hub and blades with rotation applied
    glPushMatrix()
    glRotatef(angle, 0.0, 0.0, 1.0)

    # Draw fan hub
    glColor3f(0.5, 0.5, 0.7)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.0)
    for i in range(0, 361, 10):
        rad = math.radians(i)
        glVertex2f(math.cos(rad) * 0.15, math.sin(rad) * 0.15)
    glEnd()

    # Draw 3 fan blades
    for i in range(3):
        glPushMatrix()
        glRotatef(i * 120.0, x_ax, y_ax, z_ax)
        draw_fan_blade()
        glPopMatrix()
    glPopMatrix()

    glutSwapBuffers()

def idle():
    global angle
    angle += 0.5
    if angle > 360:
        angle -= 360
    glutPostRedisplay()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-2.0, 2.0, -2.0, 2.0)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    global scale
    key = key.decode("utf-8")
    if key == 'w':
        scale += 0.1  # Scale up
    elif key == 's':
        scale = max(0.1, scale - 0.1)  # Scale down

def special_keys(key, x, y):
    global tx, ty, x_ax, y_ax, z_ax, y_rod, x_rod
    step = 0.1
    if key == GLUT_KEY_LEFT:
        tx = max(-1.5, tx - step)
        x_rod = max(-0.1, x_rod - step)
    elif key == GLUT_KEY_RIGHT:
        tx = min(tx + step, 1.5)
        x_rod = min(-0.1, x_rod + step)
    elif key == GLUT_KEY_UP:
        ty = min(1.5, ty + step)
        y_rod = max(-0.6, y_rod - step)
    elif key == GLUT_KEY_DOWN:
        ty = max(ty - step, -1.5)
        y_rod = min(0.6, y_rod+step)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"Ceiling Fan with Rod - Rotation, Translation, Scaling")

    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)

    glClearColor(1.0, 1.0, 1.0, 1.0)  # White background

    glutMainLoop()

if __name__ == "__main__":
    main()
