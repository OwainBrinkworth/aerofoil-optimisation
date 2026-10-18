# generate_naca0012_2D_prism.py
import math
import pathlib

# -----------------------------
# PARAMETERS
# -----------------------------
t = 0.12           # 12% thickness
c = 1.0            # chord
n_pts = 100        # points per surface (upper/lower)
lc_airfoil = 0.01  # mesh size near airfoil
lc_farfield = 0.5  # mesh size far-field
R_far = 30*c       # far-field box size
y_wall = 5e-5      # first prism layer height at wall
thickness_BL = 0.05 # total prism thickness (~5% chord)
growth_BL = 1.2     # growth rate for prism layers
n_layers = 15       # number of prism layers
output_file = "naca0012_2D_prism.geo"

# -----------------------------
# 1. Generate NACA0012 coordinates
# -----------------------------
xs = [0.5*(1 - math.cos(math.pi*i/n_pts)) for i in range(n_pts+1)]

points = []

# Upper surface: TE -> LE
for x in xs[::-1]:
    yt = 5 * t * c * (0.2969*math.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    points.append((x, yt))

# Lower surface: LE -> TE (excluding duplicate LE)
for x in xs[1:]:
    yt = 5 * t * c * (0.2969*math.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    points.append((x, -yt))

n_total = len(points)

# -----------------------------
# 2. Write .geo file
# -----------------------------
geo = []

geo.append("// Auto-generated NACA0012 2D mesh with boundary layer prism layers")
geo.append(f"lc_airfoil = {lc_airfoil};")
geo.append(f"lc_farfield = {lc_farfield};")
geo.append("")

# Far-field box points
geo.append(f"Point(1001) = {{-{R_far},{-R_far},0, lc_farfield}};")
geo.append(f"Point(1002) = {{{R_far},{-R_far},0, lc_farfield}};")
geo.append(f"Point(1003) = {{{R_far},{R_far},0, lc_farfield}};")
geo.append(f"Point(1004) = {{-{R_far},{R_far},0, lc_farfield}};")

# Far-field lines
geo.append("Line(1001) = {1001,1002};")
geo.append("Line(1002) = {1002,1003};")
geo.append("Line(1003) = {1003,1004};")
geo.append("Line(1004) = {1004,1001};")
geo.append("Line Loop(2001) = {1001,1002,1003,1004};")
geo.append("")

# Airfoil points
for i, (x, y) in enumerate(points):
    geo.append(f"Point({i+1}) = {{{x},{y},0, lc_airfoil}};")

# Airfoil lines connecting consecutive points
for i in range(n_total-1):
    geo.append(f"Line({i+1}) = {{{i+1},{i+2}}};")
geo.append(f"Line({n_total}) = {{{n_total},{1}}};")  # close loop

# Line loop for airfoil
line_ids = ",".join(str(i+1) for i in range(n_total))
geo.append(f"Line Loop(3001) = {{{line_ids}}};")

# Plane surface: far-field minus airfoil
geo.append("Plane Surface(4001) = {2001,3001};")
geo.append("")

# Physical groups
geo.append(f'Physical Line("airfoil") = {{1:{n_total}}};')
geo.append('Physical Line("farfield") = {1001,1002,1003,1004};')
geo.append('Physical Surface("fluid") = {4001};')
geo.append("")

# -----------------------------
# 3. Boundary layer prism layers
# -----------------------------
geo.append("// Boundary layer prism layers along the airfoil")
geo.append("Field[1] = BoundaryLayer;")
geo.append(f"Field[1].EdgesList = {{1:{n_total}}};")  # all airfoil lines
geo.append(f"Field[1].hfar = {lc_farfield};")       # far-field element size
geo.append(f"Field[1].hwall_n = {y_wall};")         # first prism layer height
geo.append(f"Field[1].thickness = {thickness_BL};") # total prism thickness
geo.append(f"Field[1].ratio = {growth_BL};")        # growth rate
geo.append(f"Field[1].Quads = 1;")                  # generate prisms along wall
geo.append("BoundaryLayerField = 1;")
geo.append("")

# -----------------------------
# 4. Generate 2D mesh
# -----------------------------
geo.append("Mesh 2;")

# -----------------------------
# 5. Write to file
# -----------------------------
path = pathlib.Path(output_file)
path.write_text("\n".join(geo))
print(f"Wrote {output_file} with {n_total} airfoil points and {n_layers} prism layers along wall.")
