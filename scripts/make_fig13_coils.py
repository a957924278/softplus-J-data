"""Create fig_05: half-period LCFS with five finite-build coil groups."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fig_common import data_path, figure_path

DEFAULT_EQ = data_path("equilibria", "coils_eq_tmp126.h5")
DEFAULT_COILS = data_path("coilsets", "tmp126_coils_n4_hard_bn_short19p2.h5")
DEFAULT_OUTPUT = figure_path("fig13_coils.png")
DEFAULT_COIL_WIDTH = 0.03
DEFAULT_COIL_HEIGHT = 0.04


def rotate_about_axis(vector, axis, angle):
    """Rotate a vector about a unit axis using Rodrigues' formula."""
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def parallel_transport(vector, tangent_from, tangent_to):
    """Apply the minimum rotation taking one unit tangent to another."""
    rotation_axis = np.cross(tangent_from, tangent_to)
    sin_angle = np.linalg.norm(rotation_axis)
    cos_angle = np.clip(np.dot(tangent_from, tangent_to), -1.0, 1.0)
    if sin_angle < 1e-12:
        transported = vector.copy()
    else:
        rotation_axis /= sin_angle
        transported = rotate_about_axis(
            vector, rotation_axis, np.arctan2(sin_angle, cos_angle)
        )
    transported -= np.dot(transported, tangent_to) * tangent_to
    return transported / np.linalg.norm(transported)


def bishop_frame(centerline):
    """Construct a periodic rotation-minimizing frame on a closed curve."""
    tangent = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    reference = np.array([0.0, 0.0, 1.0])
    normal_0 = reference - np.dot(reference, tangent[0]) * tangent[0]
    if np.linalg.norm(normal_0) < 0.1:
        reference = np.array([1.0, 0.0, 0.0])
        normal_0 = reference - np.dot(reference, tangent[0]) * tangent[0]
    normal_0 /= np.linalg.norm(normal_0)

    normal = np.empty_like(centerline)
    normal[0] = normal_0
    for idx in range(1, len(centerline)):
        normal[idx] = parallel_transport(
            normal[idx - 1], tangent[idx - 1], tangent[idx]
        )

    # Distribute the closed-loop holonomy by arclength so the seam has the
    # same small incremental twist as the rest of the rotation-minimizing frame.
    normal_at_start = parallel_transport(normal[-1], tangent[-1], tangent[0])
    closure_angle = np.arctan2(
        np.dot(tangent[0], np.cross(normal_at_start, normal_0)),
        np.dot(normal_at_start, normal_0),
    )
    segment_lengths = np.linalg.norm(
        np.roll(centerline, -1, axis=0) - centerline, axis=1
    )
    arclength_fraction = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    arclength_fraction /= np.sum(segment_lengths)
    for idx, fraction in enumerate(arclength_fraction):
        normal[idx] = rotate_about_axis(
            normal[idx], tangent[idx], fraction * closure_angle
        )

    binormal = np.cross(tangent, normal)
    binormal /= np.linalg.norm(binormal, axis=1, keepdims=True)
    return tangent, normal, binormal


def add_finite_build_coil(
    fig, coil, grid, width, height, row, col, light_direction, light_position
):
    """Add a rectangular finite-build cross-section swept along a coil."""
    import plotly.graph_objects as go

    centerline = np.asarray(coil.compute("x", grid=grid, basis="xyz")["x"], dtype=float)
    if centerline.ndim == 2 and centerline.shape[1] == 3:
        centerline = centerline
    else:
        centerline = centerline.reshape(-1, 3)

    if np.linalg.norm(centerline[-1] - centerline[0]) < 1e-10:
        centerline = centerline[:-1]
    _, normal, binormal = bishop_frame(centerline)

    half_width, half_height = width / 2.0, height / 2.0
    corners = np.array(((1, 1), (-1, 1), (-1, -1), (1, -1)), dtype=float)
    vertices = (
        centerline[:, None, :]
        + corners[None, :, 0, None] * half_width * normal[:, None, :]
        + corners[None, :, 1, None] * half_height * binormal[:, None, :]
    ).reshape(-1, 3)
    n_center = len(centerline)
    i, j, k = [], [], []
    for p in range(n_center):
        q = (p + 1) % n_center
        for m in range(4):
            n = (m + 1) % 4
            a, b = 4 * p + m, 4 * p + n
            c, d = 4 * q + m, 4 * q + n
            i.extend((a, a))
            j.extend((c, d))
            k.extend((d, b))

    # Kaleido does not reliably shade this swept mesh's shared vertices, so
    # compute diffuse face colors explicitly for consistent PNG/HTML rendering.
    triangles = vertices[np.column_stack((i, j, k))]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    light_direction = np.asarray(light_direction, dtype=float)
    light_direction /= np.linalg.norm(light_direction)
    illumination = 0.62 + 0.52 * np.maximum(face_normals @ light_direction, 0.0)
    base_color = np.array([196.0, 90.0, 37.0])  # #C45A25
    face_rgb = np.clip(illumination[:, None] * base_color, 0, 255).astype(int)
    face_colors = [f"rgb({r},{g},{b})" for r, g, b in face_rgb]

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color="#C45A25",
            facecolor=face_colors,
            opacity=1.0,
            flatshading=True,
            lighting=dict(
                ambient=1.0,
                diffuse=0.25,
                specular=0.90,
                roughness=0.22,
                fresnel=0.16,
            ),
            lightposition=dict(
                x=light_position[0], y=light_position[1], z=light_position[2]
            ),
            showscale=False,
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eq", default=str(DEFAULT_EQ), help="DESC equilibrium path.")
    parser.add_argument("--coils", default=str(DEFAULT_COILS), help="DESC coilset path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--html-output", default=None, help="Optional Plotly HTML path.")
    parser.add_argument(
        "--coil-width",
        type=float,
        default=DEFAULT_COIL_WIDTH,
        help="Rectangular coil width in meters (default: 0.03).",
    )
    parser.add_argument(
        "--coil-height",
        type=float,
        default=DEFAULT_COIL_HEIGHT,
        help="Rectangular coil height in meters (default: 0.04).",
    )
    args = parser.parse_args()

    from desc.grid import LinearGrid
    from desc.integrals import compute_B_plasma
    from desc.io import load

    eq = load(str(args.eq))
    coils = load(str(args.coils))

    # endpoint=True covers half a field period: zeta in [0, pi / NFP].
    M, N = 16, 16
    SOURCE_N = 51
    lcfs = LinearGrid(
        rho=1.0,
        M=M,
        zeta=np.linspace(0.0, np.pi / eq.NFP, 2 * N + 1),
        NFP=eq.NFP,
        endpoint=True,
    )
    coil_source = LinearGrid(N=SOURCE_N)

    # -- Compute |Bn|/|B| on LCFS --
    eq_data = eq.compute(
        ["R", "phi", "Z", "n_rho", "|e_theta x e_zeta|", "|B|"],
        grid=lcfs,
    )
    b_abs = np.asarray(eq_data["|B|"], dtype=float)
    x_rpz = np.asarray(
        [eq_data["R"], eq_data["phi"], eq_data["Z"]], dtype=float
    ).T
    b_ext = np.asarray(
        coils.compute_magnetic_field(x_rpz, basis="rpz", source_grid=coil_source),
        dtype=float,
    )
    b_ext_normal = np.sum(b_ext * np.asarray(eq_data["n_rho"], dtype=float), axis=-1)
    b_plasma_normal = np.asarray(compute_B_plasma(eq, lcfs, normal_only=True))
    bn_total = b_ext_normal + b_plasma_normal
    bn_over_b = np.abs(bn_total) / b_abs

    # Reshape to (M * NFP, N) for surface plot
    n_theta = lcfs.num_theta  # M
    n_zeta = lcfs.num_zeta    # N * NFP (with endpoint)
    R_surf = eq_data["R"].reshape(n_theta, n_zeta)
    Z_surf = eq_data["Z"].reshape(n_theta, n_zeta)
    phi_surf = eq_data["phi"].reshape(n_theta, n_zeta)
    X_surf = R_surf * np.cos(phi_surf)
    Y_surf = R_surf * np.sin(phi_surf)
    bn_surf = bn_over_b.reshape(n_theta, n_zeta)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.01,
    )

    coil_grid = LinearGrid(N=400, endpoint=True)
    cameras = (
        dict(eye=dict(x=0.92, y=0.92, z=0.58)),
        dict(eye=dict(x=-0.92, y=-0.92, z=0.58)),
    )
    coil_lights = (
        ((-0.35, -0.45, 0.82), (-3.0, -4.0, 6.0)),
        ((0.35, 0.45, 0.82), (3.0, 4.0, 6.0)),
    )
    for col, camera in enumerate(cameras, start=1):
        fig.add_trace(
            go.Surface(
                x=X_surf,
                y=Y_surf,
                z=Z_surf,
                surfacecolor=bn_surf,
                colorscale="Blues",
                opacity=1.0,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
                showscale=col == 2,
                colorbar=dict(
                    title="|B<sub>n</sub>|/|B|",
                    len=0.66,
                    thickness=14,
                    tickvals=[0, 0.005, 0.01, 0.015],
                    ticktext=["0", "5×10⁻³", "1×10⁻²", "1.5×10⁻²"],
                    title_font=dict(size=17),
                    tickfont=dict(size=14),
                ),
                showlegend=False,
                hovertemplate="|Bn|/|B|=%{surfacecolor:.2e}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        for coil in coils.coils:
            add_finite_build_coil(
                fig,
                coil,
                coil_grid,
                args.coil_width,
                args.coil_height,
                row=1,
                col=col,
                light_direction=coil_lights[col - 1][0],
                light_position=coil_lights[col - 1][1],
            )

        scene = "scene" if col == 1 else f"scene{col}"
        fig.layout[scene].update(
            domain=dict(x=[0.0, 0.50] if col == 1 else [0.47, 0.95]),
            aspectmode="data",
            camera=camera,
            xaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=""),
        )

    fig.update_layout(
        width=1200,
        height=500,
        margin=dict(l=0, r=72, t=30, b=0),
        paper_bgcolor="white",
        annotations=[
            dict(x=0.02, y=0.99, xref="paper", yref="paper", text="(a) Isometric view", showarrow=False, font=dict(size=18)),
            dict(x=0.51, y=0.99, xref="paper", yref="paper", text="(b) Opposite view", showarrow=False, font=dict(size=18)),
        ],
    )

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = figure_path(output) if not Path(output).is_absolute() else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    html_output = (
        Path(args.html_output).expanduser()
        if args.html_output
        else output.with_suffix(".html")
    )
    if not html_output.is_absolute():
        html_output = figure_path(html_output) if not Path(html_output).is_absolute() else Path(html_output)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_output), include_plotlyjs=True, full_html=True)
    print(f"saved: {html_output.resolve()}")

    try:
        fig.write_image(str(output), width=1200, height=500, scale=2)
        print(f"saved: {output.resolve()}")
    except Exception as exc:
        print(f"PNG export failed: {type(exc).__name__}: {exc}")
        print("The interactive HTML output was written successfully.")


if __name__ == "__main__":
    main()
