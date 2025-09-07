from manim import *
import numpy as np
import numpy.ma as ma
from TSPHardener.core.algorithm import choose_city_pair_to_branch
from manim.mobject.text.text_mobject import Text
from manim.mobject.text.tex_mobject import MathTex, Tex

Text.set_default(color=BLACK)
MathTex.set_default(color=BLACK)
Tex.set_default(color=BLACK)
Integer.set_default(color=BLACK)


def get_reduction_steps(matrix):
    steps = []
    mat = matrix.copy()
    lower_bound = 0
    # Row reduction
    for i in range(mat.shape[0]):
        row = mat[i, :]
        min_val = np.min(row)
        if not np.isinf(min_val) and min_val > 0:
            mat[i, :] -= min_val
            lower_bound += min_val
            steps.append((mat.copy(), ('row', i, min_val), lower_bound))
    # Column reduction
    for j in range(mat.shape[1]):
        col = mat[:, j]
        min_val = np.min(col)
        if not np.isinf(min_val) and min_val > 0:
            mat[:, j] -= min_val
            lower_bound += min_val
            steps.append((mat.copy(), ('col', j, min_val), lower_bound))
    return steps

class ReduceMatrixAnimation(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        # Example matrix
        matrix = np.array([
            [np.inf, 27, 43, 16, 30, 26],
            [7, np.inf, 16, 1, 30, 25],
            [20, 13, np.inf, 35, 5, 0],
            [21, 16, 25, np.inf, 18, 18],
            [12, 46, 27, 48, np.inf, 5],
            [23, 5, 5, 9, 5, np.inf]
        ])
        steps = get_reduction_steps(matrix)
        
        def make_table(mat):
            disp = [[("∞" if np.isinf(x) else str(int(x))) for x in row] for row in mat]
            row_labels = [MathTex(f"{i+1}", color=BLACK).scale(0.65)
                        for i in range(mat.shape[0])]
            col_labels = [MathTex(f"{j+1}", color=BLACK).scale(0.65)
                        for j in range(mat.shape[1])]
            return Table(
                disp,
                include_outer_lines=True,
                row_labels=row_labels,
                col_labels=col_labels,
            )
        
        table = make_table(matrix)
        table.scale(0.7)
        table.to_edge(UP, buff=0.1)
        self.play(Create(table))
        # self.play(*[FadeIn(label) for label in row_labels + col_labels])
        # self.wait(1)

        # Initial lower bound display
        lower_bound = 0
        lower_bound_tex = MathTex(f"\\text{{Lower Bound: }} {lower_bound}", color=BLACK)
        lower_bound_tex.next_to(table, DOWN, buff=1.1)
        self.play(FadeIn(lower_bound_tex))

        current_op = steps[0][1]
        if current_op[0] == 'row':
            desc = MathTex(f"\\text{{Subtract}}\\ {int(current_op[2])}\\ \\text{{from Row}}\\ {current_op[1]+1}", color=BLACK)
        else:
            desc = MathTex(f"\\text{{Subtract}}\\ {int(current_op[2])}\\ \\text{{from Col}}\\ {current_op[1]+1}", color=BLACK)
        desc.next_to(table, DOWN)
        self.play(FadeIn(desc))

        # Animate each step, updating lower bound
        for idx, (new_mat, op, lb) in enumerate(steps):
            if op[0] == 'row':
                highlight = table.get_rows()[op[1]]
                new_desc = MathTex(f"\\text{{Subtract}}\\ {int(op[2])}\\ \\text{{from Row}}\\ {op[1]+1}", color=BLACK)
            else:
                highlight = table.get_columns()[op[1]]
                new_desc = MathTex(f"\\text{{Subtract}}\\ {int(op[2])}\\ \\text{{from Col}}\\ {op[1]+1}", color=BLACK)
            new_desc.next_to(table, DOWN)
            self.play(highlight.animate.set_fill(RED), Transform(desc, new_desc))
            
            # Update table
            new_table = make_table(new_mat)
            new_table.scale(0.7)
            new_table.move_to(table.get_center())
            self.play(Transform(table, new_table))
            self.play(highlight.animate.set_fill(BLACK))
            self.wait(0.1)

            # Animate lower bound
            new_lb_tex = MathTex(f"\\text{{Lower Bound: }} {lb}", color=BLACK)
            new_lb_tex.next_to(table, DOWN, buff=1.1)
            self.play(Transform(lower_bound_tex, new_lb_tex))
            self.wait(0.1)

        self.wait(2)


class BranchingStepAnimation(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        matrix = np.array([
            [np.inf, 27, 43, 16, 30, 26],
            [7, np.inf, 16, 1, 30, 25],
            [20, 13, np.inf, 35, 5, 0],
            [21, 16, 25, np.inf, 18, 18],
            [12, 46, 27, 48, np.inf, 5],
            [23, 5, 5, 9, 5, np.inf]
        ])
        
        # Show matrix
        def make_table(mat):
            disp = [[("∞" if np.isinf(x) else str(int(x))) for x in row] for row in mat]
            return Table(
                disp,
                row_labels=[MathTex("R_1"), MathTex("R_2"), MathTex("R_3"), MathTex("R_4"), MathTex("R_5"), MathTex("R_6")],
                col_labels=[MathTex("C_1"), MathTex("C_2"), MathTex("C_3"), MathTex("C_4"), MathTex("C_5"), MathTex("C_6")],
                include_outer_lines=True,
                element_to_mobject_config={"color": BLACK}
            )

        table = make_table(matrix)
        table.scale(0.6)
        table.to_edge(UP, buff=0.6)
        self.play(Create(table))
        self.wait(1)
        
        # Find edge to branch
        x, y, theta = choose_city_pair_to_branch(matrix)
        edge_desc = MathTex(f"\\text{{Branch on edge }}({x+1}, {y+1})", color=BLACK)
        theta_desc = MathTex(f"\\theta = {theta:.1f}", color=BLACK)
        group = VGroup(edge_desc, theta_desc).arrange(DOWN).next_to(table, DOWN)
        self.play(FadeIn(group))
        
        # Highlight the edge
        cell = table.get_cell((x+1, y+1))
        self.play(cell.animate.set_fill(YELLOW, opacity=0.8))
        self.wait(1.5)
        
        # Optionally, show two branches
        left_label = Text("Left: Include edge", color=BLACK).scale(0.5).next_to(table, LEFT, buff=1)
        right_label = Text("Right: Exclude edge", color=BLACK).scale(0.5).next_to(table, RIGHT, buff=1)
        self.play(FadeIn(left_label), FadeIn(right_label))
        self.wait(2)

def BlackInteger(val):
    return Integer(val, color=BLACK)

class BranchingGraphScene(Scene):
    def construct(self):
        vertices = {
            "A": [-2, 1, 0],
            "B": [0, 2, 0],
            "C": [2, 1, 0],
            "D": [2, -1, 0],
            "E": [0, -2, 0],
            "F": [-2, -1, 0]
        }

        edges = [
            ("A", "B"), ("B", "C"), ("C", "D"),
            ("D", "E"), ("E", "F"), ("F", "A"),
            ("B", "E"), ("C", "F")
        ]

        g = Graph(vertices.keys(), edges,
                  layout=vertices,
                  vertex_config={"color": BLUE},
                  edge_config={"color": GREY},
                  labels=True).scale(1.5)

        self.play(Create(g))
        self.wait(1)

        # Highlight an arbitrary edge ("B", "E") and branch with binary decision
        branch_edge = g.edges[("B", "E")]

        # Decision to INCLUDE
        self.play(branch_edge.animate.set_color(GREEN), run_time=1)
        include_text = Text("Include in tour", color=GREEN, font_size=30)
        include_text.next_to(branch_edge, UP)
        self.play(Write(include_text))
        self.wait(1)

        # Revert edge color to GREY before showing the alternative
        self.play(branch_edge.animate.set_color(GREY), FadeOut(include_text), run_time=1)
        
        # Decision to EXCLUDE
        self.play(branch_edge.animate.set_color(RED), run_time=1)
        exclude_text = Text("Exclude from tour", color=RED, font_size=30)
        exclude_text.next_to(branch_edge, UP)
        self.play(Write(exclude_text))

        self.wait(2)

        # Fade out everything at end
        self.play(FadeOut(g), FadeOut(exclude_text))
        self.wait()

class ThetaCalculation(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        matrix = np.array([
            [np.inf, 11, 27, 0, 14, 10],
            [1, np.inf, 15, 0, 29, 24],
            [15, 13, np.inf, 35, 5, 0],
            [0, 0, 9, np.inf, 2, 2],
            [2, 41, 22, 43, np.inf, 0],
            [13, 0, 0, 4, 0, np.inf]
        ])

        def MT_NUMBER(x, size=34):
            # uniform MathTeX for all numeric entries
            return MathTex(str(int(x)), color=BLACK, font_size=size)

        def MT_INF(size=34):
            return MathTex(r"\infty", color=BLACK, font_size=size)

        # Cost matrix
        mat = Matrix(
            matrix, 
            element_to_mobject=lambda x: MT_INF() if x == np.inf else MT_NUMBER(x)
        )
        mat.scale(0.7).to_edge(LEFT)

        # Initial empty theta matrix (all entries are "-")
        theta_matrix = np.full(matrix.shape, None)
        theta_display_matrix = Matrix(
            [["-" for _ in range(matrix.shape[1])] for _ in range(matrix.shape[0])],
            element_to_mobject=lambda x: MathTex(str(x), color=BLACK),
        )
        theta_display_matrix.scale(0.7).to_edge(RIGHT)

        # Display both matrices
        # Labels above each matrix — use Text for both so fonts match
        mat_label = Text("Reduced Matrix", color=BLACK, font_size=28).next_to(mat, UP)
        theta_label = Text("Calculated θ Values", color=BLACK, font_size=28).next_to(theta_display_matrix, UP)

        # Two-line explainer at the very top (centered)
        top_explainer = Text(
            "Each candidate sum their lowest adjacent column and row distance\n"
            "Max θ is city pair we branch with",
            color=BLACK,
            font_size=28,
            line_spacing=0.8
        ).to_edge(UP, buff=0.3)

        # When displaying:
        self.play(Create(mat), Create(theta_display_matrix))
        # self.play(FadeIn(mat_label), FadeIn(theta_label), FadeIn(top_explainer))

        # self.wait(1)

        zeros = np.argwhere(matrix == 0)
        for zero_pos in zeros:
            x, y = zero_pos

            zero_entry = mat.get_entries()[x * matrix.shape[1] + y]
            zero_rect = SurroundingRectangle(zero_entry, color=RED)
            self.play(Create(zero_rect), run_time=0.2)

            # --- Find min in row, excluding the current zero and infs ---
            row_vals = [(j, matrix[x, j]) for j in range(matrix.shape[1]) if j != y and matrix[x, j] != np.inf]
            col_vals = [(i, matrix[i, y]) for i in range(matrix.shape[0]) if i != x and matrix[i, y] != np.inf]

            if not row_vals or not col_vals:
                theta = 0
                min_row_val, min_row_idx = None, None
                min_col_val, min_col_idx = None, None
            else:
                min_row_idx, min_row_val = min(row_vals, key=lambda t: t[1])
                min_col_idx, min_col_val = min(col_vals, key=lambda t: t[1])
                theta = min_row_val + min_col_val

            # --- Highlight only the two min values being summed ---
            highlights = []
            if row_vals:
                row_entry = mat.get_entries()[x * matrix.shape[1] + min_row_idx]
                highlights.append(SurroundingRectangle(row_entry, color=YELLOW))
            if col_vals:
                col_entry = mat.get_entries()[min_col_idx * matrix.shape[1] + y]
                highlights.append(SurroundingRectangle(col_entry, color=YELLOW))
            self.play(*(Create(h) for h in highlights), run_time=0.2)
            self.wait(0.2)

            # Update theta_display_matrix in-place
            theta_entry = theta_display_matrix.get_entries()[x * matrix.shape[1] + y]
            new_theta = MathTex(str(theta), color=BLACK, font_size=34)
            new_theta.move_to(theta_entry)
            self.play(Transform(theta_entry, new_theta), run_time=0.5)

            # Optionally, briefly highlight the theta cell
            theta_rect = SurroundingRectangle(theta_entry, color=BLUE)
            self.play(Create(theta_rect), run_time=0.3)
            self.wait(0.2)
            self.play(FadeOut(theta_rect), run_time=0.3)

            # Clean up
            self.play(*(FadeOut(h) for h in highlights), FadeOut(zero_rect), run_time=0.5)
            self.wait(0.2)

        self.wait(2)

class BranchingGraphSceneWithTree(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        # Original graph (root of tree)
        vertices = {
            "A": [-2, 1, 0],
            "B": [0, 2, 0],
            "C": [2, 1, 0],
            "D": [2, -1, 0],
            "E": [0, -2, 0],
            "F": [-2, -1, 0]
        }
        edges = [
            ("A", "B"), ("B", "C"), ("C", "D"),
            ("D", "E"), ("E", "F"), ("F", "A"),
            ("B", "E"), ("C", "F")
        ]
        graph_pos = UP * 2

        g = Graph(
            vertices.keys(), edges, layout=vertices,
            vertex_config={"color": BLUE},
            edge_config={"color": GREY}, labels=True
        ).scale(0.8).shift(graph_pos)

        self.play(Create(g))
        self.wait(0.5)

        # Layout for binary tree: root at top, two children below left/right
        child_y_shift = DOWN * 2.8
        x_offset = 3.3

        # --- LEFT CHILD: Include (B, E) edge as GREEN ---
        edge_config_include = {
            e: {"color": GREEN} if e == ("B", "E") or e == ("E", "B") else {"color": GREY}
            for e in edges
        }
        g_include = Graph(
            vertices.keys(), edges, layout=vertices,
            vertex_config={"color": BLUE},
            edge_config=edge_config_include,
            labels=True
        ).scale(0.5).shift(LEFT * x_offset + child_y_shift)

        # --- RIGHT CHILD: Exclude (B, E) edge, so just remove it ---
        edges_exclude = [e for e in edges if e != ("B", "E") and e != ("E", "B")]
        g_exclude = Graph(
            vertices.keys(), edges_exclude, layout=vertices,
            vertex_config={"color": BLUE},
            edge_config={"color": GREY},
            labels=True
        ).scale(0.5).shift(RIGHT * x_offset + child_y_shift)

        # Tree edges (arrows from root to children)
        left_arrow = Arrow(
            g.get_center() + DOWN * 1.2,
            g_include.get_center() + UP * 1.1,
            buff=0.08,
            color=BLACK
        )
        right_arrow = Arrow(
            g.get_center() + DOWN * 1.2,
            g_exclude.get_center() + UP * 1.1,
            buff=0.08,
            color=BLACK
        )

        # Labels for decisions
        include_text = Text("Include (B, E)", color=GREEN, font_size=28).next_to(left_arrow, LEFT)
        exclude_text = Text("Exclude (B, E)", color=RED, font_size=28).next_to(right_arrow, RIGHT)

        # Animate
        self.play(
            Create(left_arrow),
            Create(right_arrow),
            FadeIn(g_include),
            FadeIn(g_exclude),
            FadeIn(include_text),
            FadeIn(exclude_text),
        )
        self.wait(2)
