# https://codeloop.org/python-modern-opengl-perspective-projection/

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
from pyrr import Quaternion, Matrix33, Matrix44, Vector4
from PIL import Image

def main():
    if not glfw.init():
        return
    _width = 600
    _height = 600
    _aspect = _width / _height
    window = glfw.create_window(_width, _height, "Pyopengl Perspective Projection", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    #        positions         colors          texture coords
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
    # convert to 32bit float

    cube = np.array(cube, dtype=np.float32)

    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               8, 9, 10, 10, 11, 8,
               12, 13, 14, 14, 15, 12,
               16, 17, 18, 18, 19, 16,
               20, 21, 22, 22, 23, 20]

    indices = np.array(indices, dtype=np.uint32)

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
                gl_Position = projection * view * model * transform * vec4(position, 1.0f);
                //gl_Position = projection * view * model * vec4(position, 1.0f);
               newColor = color;
               OutTexCoords = InTexCoords;

                }
          """

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

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)

    # Create EBO
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    # get the position from  shader
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

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

    # Creating Projection Matrix
    view = Matrix44.from_translation(pyrr.Vector3([0.0, 0.0, -2.0]))
    print("Log: view \n", view)
    projection = None

    # project 1
    perspective = Matrix44.perspective_projection(80.0, _aspect, 0.1, 100.0)

    # project 2
    _max = 2
    _r = _max * _aspect
    _t = _max
    _l = -_r
    _b = -_t

    worldToCamera = Matrix44([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    def multPointMatrix(_in, _out, _M):
        # out = in * Mproj;
        # /* _in.z = 1 */
        _out.x = _in.x * _M[0][0] + _in.y * _M[1][0] + _in.z * _M[2][0] + _M[3][0]
        _out.y = _in.x * _M[0][1] + _in.y * _M[1][1] + _in.z * _M[2][1] + _M[3][1]
        _out.z = _in.x * _M[0][2] + _in.y * _M[1][2] + _in.z * _M[2][2] + _M[3][2]
        w = _in.x * _M[0][3] + _in.y * _M[1][3] + _in.z * _M[2][3] + _M[3][3]

        # normalize if w is different than 1 (convert from homogeneous to Cartesian coordinates)
        if w != 1:
            _out.x /= w
            _out.y /= w
            _out.z /= w

    class MaxVal:
        SI8 = 2 ** 7 - 1
        UI8 = 2 ** 8 - 1
        SI16 = 2 ** 15 - 1
        UI16 = 2 ** 16 - 1
        SI32 = 2 ** 31 - 1
        UI32 = 2 ** 32 - 1
        SI64 = 2 ** 63 - 1
        UI64 = 2 ** 64 - 1

    kInfinity = MaxVal.UI32

    class WorldClass:
        def __init__(self, initValue):
            self.x = initValue
            self.y = initValue
            self.z = initValue

    minWorld = WorldClass(kInfinity)
    maxWorld = WorldClass(-kInfinity)

    for i in range(0, len(cube), 8):
        if (cube[i] < minWorld.x): minWorld.x = cube[i]
        if (cube[i+1] < minWorld.y): minWorld.y = cube[i+1]
        if (cube[i+2] < minWorld.z): minWorld.z = cube[i+2]

        if (cube[i] > maxWorld.x): maxWorld.x = cube[i]
        if (cube[i+1] > maxWorld.y): maxWorld.y = cube[i+1]
        if (cube[i+2] > maxWorld.z): maxWorld.z = cube[i+2]

    print("Log: max World ", maxWorld.x, maxWorld.y, maxWorld.z)
    print("Log: min World ", minWorld.x, minWorld.y, minWorld.z)
    print("Log: -----")
    minCamera = WorldClass(0)
    maxCamera = WorldClass(0)

    print("Log: M0 ", worldToCamera[0][0],worldToCamera[0][1], worldToCamera[0][2], worldToCamera[0][3])
    print("Log: M1 ", worldToCamera[1][0],worldToCamera[1][1], worldToCamera[1][2], worldToCamera[1][3])
    print("Log: M2 ", worldToCamera[2][0],worldToCamera[2][1], worldToCamera[2][2], worldToCamera[2][3])
    print("Log: M3 ", worldToCamera[3][0],worldToCamera[3][1], worldToCamera[3][2], worldToCamera[3][3])
    print("Log: -----")
    multPointMatrix(minWorld, minCamera, worldToCamera)
    multPointMatrix(maxWorld, maxCamera, worldToCamera)

    print("Log: max Camera ", maxCamera.x, maxCamera.y, maxCamera.z)
    print("Log: min Camera ", minCamera.x, minCamera.y, minCamera.z)

    maxx = max(abs(minCamera.x), abs(maxCamera.x))
    maxy = max(abs(minCamera.y), abs(maxCamera.y))
    maxvalue = max(maxx, maxy)
    _r = maxvalue * _aspect
    _t = maxvalue
    _l = -_r
    _b = -_t

    print("Log: Camera ", _l, _r, _t, _b)

    orthogonal = Matrix44.orthogonal_projection(_l, _r, _t, _b, 0.1, 100)
    model = Matrix44.from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

    projection = perspective

    # ---------------------------------------------------------------------------
    print("Log: -----")
    print("Log: perspective \n", perspective)
    print("Log: orthogonal \n", orthogonal)
    print("Log: model \n", model)

    _v = Vector4([-0.5, 0, 0, 1])
    # view - .0, .0, -2
    # proj - 80 1.0 0.1 100
    # mode - .0, .0, .0
    #
    # 0.5: 182/300  0.5958
    # 1.0:          1.1917
    # Modify view will not effect the clip cause the geometry is already in the box with size [-1,-1,-1]~[1,1,1]
    # and only has that it's whether we see it, not whether it's clipped

    _result = projection * view * model * _v
    _comp = projection * view * model
    _x = _v.x
    _y = _v.y
    _z = _v.z
    _w = _v.w
    posX = _x * _comp.m11 + _y * _comp.m21 + _z * _comp.m31 + _w * _comp.m41
    posY = _x * _comp.m12 + _y * _comp.m22 + _z * _comp.m32 + _w * _comp.m42
    posZ = _x * _comp.m13 + _y * _comp.m23 + _z * _comp.m33 + _w * _comp.m43
    posW = _x * _comp.m14 + _y * _comp.m24 + _z * _comp.m34 + _w * _comp.m44
    print("Log: -----")
    print("Log: V ", _v)
    print("Log: CLIP X Y Z W", posX, posY, posZ, posW)
    print("Log: -----")
    print("Log: M Result \n", model * _v)
    print("Log: MV Result \n", view * model * _v)
    print("Log: MVP Result \n", _result)
    print("Log: -----")
    print("Log: NDC X Y Z", posX/posW, posY/posW, posZ/posW)

    # ---------------------------------------------------------------------------

    # projection = perspective

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.05 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.08 * glfw.get_time())

        transformLoc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

        # Draw Cube

        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()