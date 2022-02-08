import math

from manimlib import *


class LinearTransformationSceneExample(Scene):
    def construct(self):
        self.wait()
        intro_words = Text("""""")
        intro_words.to_edge(UP)

        # Square
        square = Square()
        square.set_fill(BLUE_E, 1)
        self.add(square)

        # Linear transform
        grid = NumberPlane((-10, 10), (-10, 10))

        # i hat and j hat
        vec_1 = Vector([1, 0])
        vec_2 = Vector([0, 1])
        vec_1.set_color(color=RED)
        vec_2.set_color(color=GREEN)

        grid.add(vec_1, vec_2)
        grid.add(square)

        stand = [[1, 0],
                 [0, 1]]

        # Shear
        # matrix = [[1, 1],
        #           [0, 1]]

        # Reflection
        # matrix = [[-1, 0],
        #           [0, 1]]

        # Uniform scale
        # matrix = [[2, 0],
        #           [0, 2]]

        # Non-Uniform scale
        # matrix = [[2, 0],
        #           [0, 1]]

        angle = math.radians(90.0)
        matrix = [[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]]
        eq1 = MTexText("$\\left [ \\begin{matrix} cos\\theta & -sin\\theta \\\\ sin\\theta & cos\\theta "
                       "\\end{matrix} \\right ]$")
        eq2 = MTexText("$\\theta=90^{\\circ}$")

        linear_transform_words = VGroup(
            Text("Matrix from"),
            IntegerMatrix(stand, include_background_rectangle=True),
            Text("to"),
            IntegerMatrix(matrix, include_background_rectangle=True),
            eq1,
            eq2
        )
        linear_transform_words.arrange(RIGHT)
        linear_transform_words.to_edge(UP)
        linear_transform_words.set_stroke(BLACK, 10, background=True)

        self.play(
            ShowCreation(grid),
            FadeTransform(intro_words, linear_transform_words)
        )

        self.wait()
        self.play(grid.animate.apply_matrix(matrix), run_time=3)
        self.wait()
