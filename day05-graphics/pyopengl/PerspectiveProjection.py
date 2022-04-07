# https://codeloop.org/python-modern-opengl-perspective-projection/
import math

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
from pyrr import Quaternion, Matrix33, Matrix44, Vector4
from PIL import Image


_INTER = 0


def orthoMatrix(b, t, l, r, n, f, M):
    M[0][0] = 2 / (r - l)
    M[0][1] = 0
    M[0][2] = 0
    M[0][3] = 0

    M[1][0] = 0
    M[1][1] = 2 / (t - b)
    M[1][2] = 0
    M[1][3] = 0

    M[2][0] = 0
    M[2][1] = 0
    M[2][2] = -2 / (f - n)
    M[2][3] = 0

    M[3][0] = -(r + l) / (r - l)
    M[3][1] = -(t + b) / (t - b)
    M[3][2] = -(f + n) / (f - n)
    M[3][3] = 1


def perspMatrix(fov, aspect, n, f, M):
    t, b, l, r = fovConvert(fov, aspect, n)

    M[0][0] = 2 * n / (r - l)
    M[0][1] = 0
    M[0][2] = 0
    M[0][3] = 0

    M[1][0] = 0
    M[1][1] = 2 * n / (t - b)
    M[1][2] = 0
    M[1][3] = 0

    M[2][0] = (r + l) / (r - l)
    M[2][1] = (t + b) / (t - b)
    M[2][2] = - (f + n) / (f - n)
    M[2][3] = -1

    M[3][0] = 0
    M[3][1] = 0
    M[3][2] = -(2 * f * n) / (f - n)
    M[3][3] = 0


def multPointMatrix(_in, _out, _M):
    # out = in * Mproj;
    # /* _in.z = 1 */
    _out.x = _in.x * _M[0][0] + _in.y * _M[1][0] + _in.z * _M[2][0] + _M[3][0]
    _out.y = _in.x * _M[0][1] + _in.y * _M[1][1] + _in.z * _M[2][1] + _M[3][1]
    _out.z = _in.x * _M[0][2] + _in.y * _M[1][2] + _in.z * _M[2][2] + _M[3][2]
    w = _in.x * _M[0][3] + _in.y * _M[1][3] + _in.z * _M[2][3] + _M[3][3]

    # normalize if w is different than 1 (convert from homogeneous to Cartesian coordinates)
    if w != 1:
        print("normalizing")
        _out.x /= w
        _out.y /= w
        _out.z /= w


def fovConvert(angleOfView, imageAspectRatio, n):
    t = math.tan(angleOfView * 0.5 * math.pi / 180) * n
    b = -t
    r = imageAspectRatio * t
    l = imageAspectRatio * (-t)  # or simply l = -r
    return t, b, l, r


def createOrthogonalMatrix(aspect, cube, n, f):
    # --------------------------------------------------------------------------
    # follow
    #   https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/orthographic-projection-matrix
    #       Text Program

    _emptyOrtho = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    worldToCamera = Matrix44([  # for ortho
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    # Construct two opposite maxima and minima
    kInfinity = MaxVal.UI32
    minWorld = WorldClass(kInfinity)
    maxWorld = WorldClass(-kInfinity)

    for i in range(0, len(cube), 8):
        if (cube[i] < minWorld.x): minWorld.x = cube[i]
        if (cube[i + 1] < minWorld.y): minWorld.y = cube[i + 1]
        if (cube[i + 2] < minWorld.z): minWorld.z = cube[i + 2]

        if (cube[i] > maxWorld.x): maxWorld.x = cube[i]
        if (cube[i + 1] > maxWorld.y): maxWorld.y = cube[i + 1]
        if (cube[i + 2] > maxWorld.z): maxWorld.z = cube[i + 2]

    print("Log: max World ", maxWorld.x, maxWorld.y, maxWorld.z)
    print("Log: min World ", minWorld.x, minWorld.y, minWorld.z)

    minCamera = WorldClass2(minWorld)
    maxCamera = WorldClass2(maxWorld)

    # Multiply camera matrix
    # multPointMatrix(minWorld, minCamera, worldToCamera)
    # multPointMatrix(maxWorld, maxCamera, worldToCamera)

    # setup `r t b l` values
    maxx = max(abs(minCamera.x), abs(maxCamera.x))
    maxy = max(abs(minCamera.y), abs(maxCamera.y))
    maxvalue = max(maxx, maxy)
    _r = maxvalue * aspect
    _t = maxvalue
    _l = -_r
    _b = -_t
    print("Log: left right top bottom ", _l, _r, _t, _b)

    # Use pyrr library to create Ortho-Matrix
    orthogonal = Matrix44.orthogonal_projection(_l, _r, _t, _b, n, f)

    # Use our self method to create Ortho-Matrix
    orthoMatrix(_b, _t, _l, _r, 0.1, 100, _emptyOrtho)
    orthogonalTest = Matrix44(_emptyOrtho)

    print("Log: testOrtho", _emptyOrtho)
    print("Log: orthogonalTest \n", orthogonalTest)

    return orthogonal, orthogonalTest


def createPerspMatrix(fov, aspect, n, f):
    _emptyPersp = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    # Use our self method to create Persp-Matrix
    perspMatrix(fov, aspect, n, f, _emptyPersp)
    perspectiveTest = Matrix44(_emptyPersp)
    print("Log: testPersp", _emptyPersp)
    print("Log: perspectiveTest \n", perspectiveTest)

    # Use pyrr library to create Persp-Matrix
    perspective = Matrix44.perspective_projection(fov, aspect, n, f)

    return perspective, perspectiveTest


class MaxVal:
    SI8 = 2 ** 7 - 1
    UI8 = 2 ** 8 - 1
    SI16 = 2 ** 15 - 1
    UI16 = 2 ** 16 - 1
    SI32 = 2 ** 31 - 1
    UI32 = 2 ** 32 - 1
    SI64 = 2 ** 63 - 1
    UI64 = 2 ** 64 - 1


class WorldClass:
    def __init__(self, initValue):
        self.x = initValue
        self.y = initValue
        self.z = initValue


class WorldClass2:
    def __init__(self, initValue):
        self.x = initValue.x
        self.y = initValue.y
        self.z = initValue.z


def main():
    # init glfw
    if not glfw.init():
        return

    if _INTER == 0:
        _width = 600
        _height = 600
    elif _INTER == 1:
        _width = 640
        _height = 480
    _aspect = _width / _height

    # create our window
    window = glfw.create_window(_width, _height, "Perspective Projection", None, None)

    if not window:
        glfw.terminate()
        return

    # set OpenGL context
    glfw.make_context_current(window)

    # prepare our cube render data
    #        positions         colors       texture coords
    cube = [-0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

            -0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, -0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

            0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

            -0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
            -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

            -0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            -0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

            0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0]

    # convert our cube data to 32bit float
    cube = np.array(cube, dtype=np.float32)

    # prepare our render index (The order in which vertices are rendered)
    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               8, 9, 10, 10, 11, 8,
               12, 13, 14, 14, 15, 12,
               16, 17, 18, 18, 19, 16,
               20, 21, 22, 22, 23, 20]

    indices = np.array(indices, dtype=np.uint32)

    # prepare our vertex-shader
    VERTEX_SHADER = """

        #version 330
        
        in vec3 position;
        in vec3 color;
        in vec2 InTexCoords;
        
        out vec3 newColor;
        out vec2 OutTexCoords;
        
        uniform mat4 transform; 
        
        uniform mat4 view;
        uniform mat4 model;
        uniform mat4 projection;
        
        void main() {
        // Animation
        //gl_Position = projection * view * model * transform * vec4(position, 1.0f);
        gl_Position = projection * view * model * vec4(position, 1.0f);
        newColor = color;
        OutTexCoords = InTexCoords;
        
        }
    """

    # prepare our fragment-shader
    FRAGMENT_SHADER = """
        #version 330
        
        in vec3 newColor;
        in vec2 OutTexCoords;
        
        out vec4 outColor;
        uniform sampler2D samplerTex;
        
        void main() {
            outColor = texture(samplerTex, OutTexCoords);
        }
    """

    # Compile The Program and shaders
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

    # Create Vertex-Buffer-Objects
    # for more: https://learnopengl.com/Getting-started/Hello-Triangle
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)

    # Create Element-Buffer-Objects
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    # get the position/color/coord from shader, The order is the same as when we defined cube
    position = 0
    glBindAttribLocation(shader, position, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = 1
    glBindAttribLocation(shader, color, 'color')
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    texCoords = 2
    glBindAttribLocation(shader, texCoords, 'texCoords')
    glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(24))
    glEnableVertexAttribArray(texCoords)

    # prepare our texture buffer
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # load image
    image = Image.open("wood.jpg")
    img_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glEnable(GL_TEXTURE_2D)

    glUseProgram(shader)

    # set background color
    glClearColor(0.125, 0.125, 0.125, 1.0)
    glEnable(GL_DEPTH_TEST)

    # Creating Projection Matrix
    n = 0.01
    f = 100.0
    fov = 90.0

    # create perspective project
    perspective, perspectiveTest = createPerspMatrix(fov, _aspect, n, f)

    # create orthogonal project
    orthogonal, orthogonalTest = createOrthogonalMatrix(_aspect, cube, n, f)

    # for tweak to see different effect by apply diff-project
    #   edit this [orthogonal, orthogonalTest, perspective, perspectiveTest]
    projection = perspectiveTest

    # ---------------------------------------------------------------------------
    # ------------------------------ Our test -----------------------------------
    model = Matrix44.from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))
    view = Matrix44.from_translation(pyrr.Vector3([0.0, 0.0, -1.0]))
    screen = Matrix44([
        [_width/2.0, 0, 0, _width/2.0],
        [0, _height/2.0, 0, _height/2.0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]])

    screen = screen.transpose()

    print("Log: -----")
    print("Log: view \n", view)
    print("Log: perspective \n", perspective)
    print("Log: orthogonal \n", orthogonal)
    print("Log: model \n", model)
    print("Log: screen \n", screen)

    original_v = Vector4([0.5, 0.5, 0.5, 1])

    mvp_matrix = projection * view * model
    _x = original_v.x
    _y = original_v.y
    _z = original_v.z
    _w = original_v.w
    clip_x = _x * mvp_matrix.m11 + _y * mvp_matrix.m21 + _z * mvp_matrix.m31 + _w * mvp_matrix.m41
    clip_y = _x * mvp_matrix.m12 + _y * mvp_matrix.m22 + _z * mvp_matrix.m32 + _w * mvp_matrix.m42
    clip_z = _x * mvp_matrix.m13 + _y * mvp_matrix.m23 + _z * mvp_matrix.m33 + _w * mvp_matrix.m43
    clip_w = _x * mvp_matrix.m14 + _y * mvp_matrix.m24 + _z * mvp_matrix.m34 + _w * mvp_matrix.m44

    # notice: we must set the w value to 1 after we already divided by w
    ndc_v = Vector4([clip_x/clip_w, clip_y/clip_w, clip_z/clip_w, 1])
    print("Log: -----")
    print("Log: V \n", original_v)
    print("Log: CLIP \n", clip_x, clip_y, clip_z, clip_w)
    print("Log: -----")
    print("Log: M Result \n", model * original_v)
    print("Log: MV Result \n", view * model * original_v)
    print("Log: MVP Result \n", projection * view * model * original_v)  # mvp-result must be same as clip-result
    print("Log: NDC (divide by w) Result \n", ndc_v)
    print("Log: Screen Result \n", screen * ndc_v)
    print("Log: -----")
    # ---------------------------------------------------------------------------

    # setup MVP matrix to shader
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    while not glfw.window_should_close(window):
        # handle pending events
        glfw.poll_events()

        # clear buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.05 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.08 * glfw.get_time())

        # Animation cube
        transformLoc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

        # Draw Cube
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
