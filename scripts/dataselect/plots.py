"""Plotly helpers specific to the data-selective-training page.

Kept small: the page reuses the weakspot plotting utilities
(``plot_data_overview``, ``plot_ground_truth_error``, ``plot_comparison``, …)
for everything those already cover, and only adds the two views that are new
here — the true landscape and the Gaussian selection kernel.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _circle_xy(center, radius, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    return center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)


def plot_landscape(xx, yy, true_surface, center, radius,
                   title="Ground-truth landscape f(x₁, x₂)"):
    """Filled contour of the noiseless target with the induced weakspot overlaid."""
    Z = np.asarray(true_surface).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z, colorscale="Viridis",
        colorbar=dict(title="f(x₁, x₂)"), contours=dict(showlines=False),
        opacity=0.9,
    ))
    if radius and radius > 0:
        cx, cy = _circle_xy(center, radius)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced weakspot",
        ))
    fig.update_layout(
        title=title, xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=460, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_selection(X_pool, weights, X_sel, sel_center, sel_sigma,
                   induced_center, induced_radius,
                   title="Weakspot-guided data selection", show_kernel=True):
    """Candidate pool with the chosen (newly selected) points highlighted.

    ``show_kernel=True`` colours the pool by Gaussian selection weight and draws
    the 1σ / 2σ selection rings around the detected weakspot centre (weakspot-
    guided view). ``show_kernel=False`` draws a neutral pool with no kernel —
    used for the random baseline, where selection ignores the weakspot.
    """
    fig = go.Figure()

    # Candidate pool — coloured by weight (guided) or neutral grey (baseline)
    if show_kernel and weights is not None:
        pool_marker = dict(color=weights, colorscale="Blues", size=5,
                           colorbar=dict(title="Selection<br>weight"),
                           line=dict(width=0.3, color="lightgray"))
    else:
        pool_marker = dict(color="#c7c7c7", size=5,
                           line=dict(width=0.3, color="lightgray"))
    fig.add_trace(go.Scatter(
        x=X_pool[:, 0], y=X_pool[:, 1], mode="markers",
        marker=pool_marker, name="Candidate pool (available)", opacity=0.6,
    ))

    # Selected points (the new datapoints that get added to the training set)
    if X_sel is not None and len(X_sel) > 0:
        fig.add_trace(go.Scatter(
            x=X_sel[:, 0], y=X_sel[:, 1], mode="markers",
            marker=dict(color="#d62728", size=7, symbol="circle",
                        line=dict(width=0.8, color="black")),
            name=f"Newly selected ({len(X_sel)})",
        ))

    # Selection kernel rings (1σ, 2σ) around the detected weakspot centroid
    if show_kernel and sel_center is not None and sel_sigma is not None:
        for k, dash in ((1, "dot"), (2, "dash")):
            rx, ry = _circle_xy(sel_center, k * sel_sigma)
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines",
                line=dict(color="#7f00ff", width=1.6, dash=dash),
                name=f"Selection {k}σ",
            ))
        fig.add_trace(go.Scatter(
            x=[sel_center[0]], y=[sel_center[1]], mode="markers",
            marker=dict(color="#7f00ff", size=11, symbol="x",
                        line=dict(width=1.5, color="black")),
            name="Detected weakspot centre",
        ))

    # Induced weakspot for reference
    if induced_radius and induced_radius > 0:
        cx, cy = _circle_xy(induced_center, induced_radius)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced weakspot",
        ))

    fig.update_layout(
        title=title, xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=520, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_training_points(X_train, y_train, X_sel, center, radius,
                         title="Training set"):
    """Scatter of the training set (optionally with newly added selected points)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_train[:, 0], y=X_train[:, 1], mode="markers",
        marker=dict(color=y_train, colorscale="Viridis", size=5,
                    colorbar=dict(title="y (noisy)"),
                    line=dict(width=0.3, color="lightgray")),
        name="Training points", opacity=0.8,
    ))
    if X_sel is not None and len(X_sel) > 0:
        fig.add_trace(go.Scatter(
            x=X_sel[:, 0], y=X_sel[:, 1], mode="markers",
            marker=dict(color="#d62728", size=7, symbol="diamond",
                        line=dict(width=0.6, color="black")),
            name=f"Newly selected points ({len(X_sel)})",
        ))
    if radius and radius > 0:
        cx, cy = _circle_xy(center, radius)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced weakspot",
        ))
    fig.update_layout(
        title=title, xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=480, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig

# Both landscape views are read *through* their colour scale (dark is low error; blue
# is error removed), and Streamlit's chart theme rewrites colour scales - so the page
# renders them with ``theme=None``. That in turn means the figure supplies its own
# chrome, and it has to sit on a page that may be light or dark: hence transparent
# backgrounds and a mid grey that stays legible either way.
_GREY = "#8a8f98"
_GRID = "rgba(136,136,136,0.28)"


def _layout(fig, title, height):
    """Shared chrome for the two landscape views: square, theme-agnostic, legend out."""
    # These panels sit three to a row in Streamlit columns of unpredictable width, so
    # the unit square cannot be kept exactly square without either padding the range
    # or collapsing the plot area. It is left to fill its panel; the height below and
    # the shared colour bar (drawn once per row) keep the distortion mild.
    axis = dict(range=[0, 1], showgrid=True, gridcolor=_GRID,
                zeroline=False, linecolor=_GRID, ticks="outside", tickcolor=_GRID)
    fig.update_layout(
        title=title, template="plotly_white", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_GREY),
        margin=dict(l=50, r=10, t=50, b=60),
        xaxis=dict(title="x₁ (norm)", **axis),
        yaxis=dict(title="x₂ (norm)", **axis),
        # Below the plot rather than inside it: these panels are shown three to a row
        # and an inset legend covers the corner of the map it sits on.
        legend=dict(orientation="h", x=0, y=-0.18, xanchor="left", yanchor="top"),
    )
    return fig


def plot_error_landscape(xx, yy, Z, center, radius, title="Error landscape",
                         zmin=None, zmax=None, height=360, colorbar_title=None,
                         showscale=True):
    """Dense-grid absolute-error landscape of one model over the unit square.

    Unlike ``plot_ground_truth_error`` (a scatter of the evaluation points) this is
    the model's error evaluated on a regular grid against the noiseless ground truth,
    so the three landscapes of a round can be put side by side and subtracted. Pass a
    common ``zmin``/``zmax`` to all three so they share one colour scale.
    """
    fig = go.Figure(go.Heatmap(
        x=xx[0], y=yy[:, 0], z=np.asarray(Z).reshape(xx.shape),
        colorscale="Inferno", zmin=zmin, zmax=zmax, zsmooth="best",
        showscale=showscale, colorbar=dict(title=colorbar_title or "|ŷ − f|"),
    ))
    if radius and radius > 0:
        cx, cy = _circle_xy(center, radius)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="Induced weakspot",
                                 line=dict(color="red", width=2, dash="dash")))
    _layout(fig, title, height)
    return fig


def plot_landscape_diff(xx, yy, D, center, radius, title="Δ error",
                        zabs=None, height=360, ellipse=None, showscale=True):
    """Difference of two error landscapes on a diverging scale centred at zero.

    Blue = error removed, red = error added. ``zabs`` fixes the symmetric limit so
    several difference maps can share one scale; ``ellipse`` optionally overlays the
    detected weakspot extent.
    """
    D = np.asarray(D).reshape(xx.shape)
    z = float(zabs) if zabs else float(np.percentile(np.abs(D), 99)) or 1e-6
    fig = go.Figure(go.Heatmap(
        x=xx[0], y=yy[:, 0], z=D, colorscale="RdBu", reversescale=True,
        zmin=-z, zmax=z, zmid=0, zsmooth="best",
        showscale=showscale, colorbar=dict(title="Δ error"),
    ))
    if radius and radius > 0:
        cx, cy = _circle_xy(center, radius)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="Induced weakspot",
                                 line=dict(color="red", width=2, dash="dash")))
    if ellipse is not None:
        fig.add_trace(go.Scatter(
            x=ellipse["ellipse_x"], y=ellipse["ellipse_y"], mode="lines",
            name="Detected weakspot", line=dict(color="#00d0ff", width=2)))
    _layout(fig, title, height)
    return fig
