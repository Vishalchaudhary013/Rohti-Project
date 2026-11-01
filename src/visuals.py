"""Chart builders: pure functions returning Plotly figures, plus Matplotlib PNG exports.

One colour system for every chart (tokens in :mod:`src.config`):
pass = blue, fail = orange with a diagonal hatch and a text label (never colour alone),
correlations use a red <-> grey <-> blue diverging scale centred on zero.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import analytics, config

PERCENT = ".1%"


# ---------------------------------------------------------------------------
# Labels and theme
# ---------------------------------------------------------------------------
def value_label(column: str, value: object) -> str:
    """Return the display label for a coded value of ``column``."""
    if column == config.SUBJECT_COL:
        return config.SUBJECT_LABELS.get(str(value), str(value))
    return config.VALUE_LABELS.get(column, {}).get(value, str(value))


def axis_title(column: str) -> str:
    """Return the axis title for ``column``."""
    return config.AXIS_TITLES.get(column, column)


def _style(fig: go.Figure, x_title: str = "", y_title: str = "", height: int = 380) -> go.Figure:
    """Apply the shared flat theme: surfaces, hairline grid, readable fonts."""
    fig.update_layout(
        height=height,
        margin={"l": 64, "r": 24, "t": 24, "b": 56},
        paper_bgcolor=config.SURFACE,
        plot_bgcolor=config.SURFACE,
        font={
            "family": config.FONT_FAMILY,
            "size": config.BASE_FONT_SIZE - 1,
            "color": config.INK_SECONDARY,
        },
        hoverlabel={
            "bgcolor": config.SURFACE,
            "bordercolor": config.BORDER,
            "font": {"color": config.INK_PRIMARY, "family": config.FONT_FAMILY},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "title": None,
            "font": {"color": config.INK_PRIMARY},
        },
        bargap=0.25,
    )
    axis = {
        "gridcolor": config.GRIDLINE,
        "linecolor": config.AXIS_LINE,
        "zeroline": False,
        "ticks": "",
        "tickfont": {"color": config.INK_MUTED},
        "title_font": {"color": config.INK_SECONDARY},
    }
    fig.update_xaxes(title_text=x_title, showgrid=False, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


def _tick_with_n(column: str, levels: pd.Series, counts: pd.Series) -> list[str]:
    """Return tick labels of the form ``"Label<br>n=123"``."""
    return [
        f"{value_label(column, v)}<br><span style='font-size:11px'>n={int(n):,}</span>"
        for v, n in zip(levels, counts, strict=True)
    ]


def _outcome_bar_style(outcome: str) -> dict[str, object]:
    """Marker settings for a pass/fail bar trace (colour + pattern)."""
    return {
        "color": config.OUTCOME_COLORS[outcome],
        "line": {"color": config.SURFACE, "width": 1},
        "pattern": {
            "shape": config.OUTCOME_PATTERNS[outcome],
            "fgcolor": config.SURFACE,
            "size": 6,
            "solidity": 0.25,
            "fillmode": "overlay",
        },
    }


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------
def g3_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of final grades, with pass and fail bars distinguished by colour and hatch."""
    grades = np.arange(config.GRADE_MIN, config.GRADE_MAX + 1)
    counts = df[config.TARGET_SOURCE].value_counts().reindex(grades, fill_value=0)
    fig = go.Figure()
    for outcome, mask in (
        (config.FAIL_LABEL, grades < config.PASS_THRESHOLD),
        (config.PASS_LABEL, grades >= config.PASS_THRESHOLD),
    ):
        fig.add_bar(
            x=grades[mask],
            y=counts.to_numpy()[mask],
            name=outcome,
            marker=_outcome_bar_style(outcome),
            hovertemplate=f"G3 = %{{x}}<br>%{{y}} students<br>{outcome}<extra></extra>",
        )
    fig.add_vline(
        x=config.PASS_THRESHOLD - 0.5,
        line={"color": config.INK_MUTED, "dash": "dot", "width": 1},
        annotation_text="Pass mark (10)",
        annotation_font={"color": config.INK_SECONDARY, "size": 11},
    )
    fig.update_layout(bargap=0.08)
    fig.update_xaxes(dtick=1, range=[-0.6, 20.6])
    return _style(fig, axis_title("G3"), "Number of students")


def pass_rate_bar(rates: pd.DataFrame, by: str, height: int = 340) -> go.Figure:
    """Pass rate per group (output of :func:`analytics.pass_rate_by`), with group sizes."""
    fig = go.Figure(
        go.Bar(
            x=list(range(len(rates))),
            y=rates["pass_rate"],
            marker={"color": config.PASS_COLOR},
            text=[f"{r:.1%}" if pd.notna(r) else "—" for r in rates["pass_rate"]],
            textposition="outside",
            textfont={"color": config.INK_PRIMARY},
            customdata=np.stack([rates["n"], rates["n_pass"]], axis=1),
            hovertemplate="Pass rate %{y:.1%}<br>%{customdata[1]} of %{customdata[0]} "
            "students passed<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(rates))),
        ticktext=_tick_with_n(by, rates[by], rates["n"]),
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.12])
    return _style(fig, axis_title(by), "Pass rate", height)


def mean_bar(means: pd.DataFrame, by: str, height: int = 340) -> go.Figure:
    """Mean G3 per group (output of :func:`analytics.mean_by`) with ±1 SE bars and sizes."""
    fig = go.Figure(
        go.Bar(
            x=list(range(len(means))),
            y=means["mean"],
            marker={"color": config.PASS_COLOR},
            error_y={
                "type": "data",
                "array": means["sem"].fillna(0),
                "color": config.INK_MUTED,
                "thickness": 1,
                "width": 4,
            },
            text=[f"{m:.1f}" if pd.notna(m) else "—" for m in means["mean"]],
            textposition="inside",
            insidetextanchor="start",
            textfont={"color": "#ffffff"},
            customdata=np.stack([means["n"], means["sem"].fillna(0)], axis=1),
            hovertemplate="Mean G3 %{y:.2f} (±%{customdata[1]:.2f} SE)<br>"
            "n = %{customdata[0]}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(means))),
        ticktext=_tick_with_n(by, means[by], means["n"]),
    )
    fig.update_yaxes(range=[0, config.GRADE_MAX])
    return _style(fig, axis_title(by), "Mean final grade (G3)", height)


def g3_box(df: pd.DataFrame, by: str, height: int = 360) -> go.Figure:
    """Box plots of G3 per level of ``by`` in fixed categorical colour order."""
    fig = go.Figure()
    levels = sorted(df[by].dropna().unique().tolist())
    for i, level in enumerate(levels):
        values = df.loc[df[by] == level, config.TARGET_SOURCE]
        color = config.SERIES_SEQUENCE[i % len(config.SERIES_SEQUENCE)]
        fig.add_box(
            y=values,
            name=f"{value_label(by, level)} (n={len(values):,})",
            marker={"color": color, "size": 5},
            line={"color": color, "width": 1.5},
            fillcolor="rgba(0,0,0,0)",
            boxmean=True,
            hoverinfo="y",
        )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[-0.5, config.GRADE_MAX + 0.5])
    return _style(fig, axis_title(by), axis_title("G3"), height)


def g2_g3_scatter(df: pd.DataFrame, seed: int = config.RANDOM_STATE) -> go.Figure:
    """G2 vs G3 scatter. Points are jittered ±0.25 so stacked students stay visible.

    Fail points use an ``x`` symbol as well as colour; hover shows exact grades and how
    many students share them.
    """
    rng = np.random.default_rng(seed)
    data = df[["G2", "G3", config.PASS_COL]].copy()
    data["n_same"] = analytics.point_counts(df, "G2", "G3").to_numpy()
    fig = go.Figure()
    for outcome, flag, symbol in (
        (config.PASS_LABEL, 1, "circle"),
        (config.FAIL_LABEL, 0, "x-thin-open"),
    ):
        part = data[data[config.PASS_COL] == flag]
        fig.add_scatter(
            x=part["G2"] + rng.uniform(-0.25, 0.25, len(part)),
            y=part["G3"] + rng.uniform(-0.25, 0.25, len(part)),
            mode="markers",
            name=outcome,
            marker={
                "color": config.OUTCOME_COLORS[outcome],
                "symbol": symbol,
                "size": 8,
                "opacity": 0.65,
                "line": {"width": 1.5, "color": config.OUTCOME_COLORS[outcome]}
                if flag == 0
                else {"width": 0.5, "color": config.SURFACE},
            },
            customdata=part[["G2", "G3", "n_same"]].to_numpy(),
            hovertemplate="G2 = %{customdata[0]}, G3 = %{customdata[1]}<br>"
            "%{customdata[2]} student(s) at this point<extra>" + outcome + "</extra>",
        )
    fig.add_hline(
        y=config.PASS_THRESHOLD - 0.5,
        line={"color": config.INK_MUTED, "dash": "dot", "width": 1},
        annotation_text="Pass mark",
        annotation_position="bottom right",
        annotation_font={"color": config.INK_SECONDARY, "size": 11},
    )
    fig.update_xaxes(range=[-0.8, 20.8], dtick=2)
    fig.update_yaxes(range=[-0.8, 20.8], dtick=2)
    return _style(fig, axis_title("G2"), axis_title("G3"), 460)


def correlation_bar(corr: pd.Series, target: str = config.TARGET_SOURCE) -> go.Figure:
    """Horizontal bars of each attribute's correlation with ``target`` (sorted)."""
    neg, pos = config.DIVERGING_SCALE[0][1], config.DIVERGING_SCALE[-1][1]
    fig = go.Figure(
        go.Bar(
            x=corr.to_numpy(),
            y=corr.index,
            orientation="h",
            marker={"color": [pos if v >= 0 else neg for v in corr]},
            text=[f"{v:+.2f}" for v in corr],
            textposition="outside",
            textfont={"color": config.INK_PRIMARY, "size": 11},
            hovertemplate="%{y}: r = %{x:+.3f}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(
        range=[-1, 1], dtick=0.25, showgrid=True, zeroline=True, zerolinecolor=config.AXIS_LINE
    )
    fig.update_layout(bargap=0.35)
    return _style(fig, f"Pearson correlation with {target}", "", max(320, 24 * len(corr) + 90))


def correlation_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Correlation heatmap on a diverging red–grey–blue scale centred on 0."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[list(s) for s in config.DIVERGING_SCALE],
            text=matrix.to_numpy(),
            texttemplate="%{text:.2f}",
            textfont={"size": 9},
            xgap=2,
            ygap=2,
            hovertemplate="%{y} × %{x}: r = %{z:+.3f}<extra></extra>",
            colorbar={"title": {"text": "r"}, "thickness": 10, "outlinewidth": 0},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45)
    side = max(420, 30 * len(matrix) + 140)
    return _style(fig, "", "", side)


# ---------------------------------------------------------------------------
# Static (Matplotlib) exports — 300 dpi PNGs for reports
# ---------------------------------------------------------------------------
FIGURE_CATALOG: dict[str, str] = {
    "01_g3_distribution.png": "Distribution of final grades (G3), pass vs fail",
    "02_pass_rate_by_subject.png": "Pass rate by subject, with group sizes",
    "03_pass_rate_by_school.png": "Pass rate by school, with group sizes",
    "04_mean_g3_by_studytime.png": "Mean G3 by weekly study time (±1 SE)",
    "05_mean_g3_by_failures.png": "Mean G3 by number of past failures (±1 SE)",
    "06_pass_rate_by_absence_band.png": "Pass rate by absence band",
    "07_g3_box_by_subject.png": "G3 box plots by subject",
    "08_g3_box_by_higher.png": "G3 box plots by higher-education aspiration",
    "09_g2_vs_g3_scatter.png": "Second-period vs final grade (jittered)",
    "10_correlation_with_g3.png": "Correlation of numeric attributes with G3",
    "11_correlation_heatmap.png": "Correlation heatmap of numeric attributes",
    "12_pass_fail_donut.png": "Share of passing vs failing records",
}


def _mpl():
    """Import Matplotlib with a non-interactive backend and apply the shared style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": config.AXIS_LINE,
            "axes.labelcolor": config.INK_SECONDARY,
            "xtick.color": config.INK_MUTED,
            "ytick.color": config.INK_MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": config.GRIDLINE,
            "grid.linewidth": 0.6,
            "axes.facecolor": config.SURFACE,
            "figure.facecolor": config.SURFACE,
            "legend.frameon": False,
        }
    )
    return plt


def _static_rate(ax, rates: pd.DataFrame, by: str) -> None:
    labels = [f"{value_label(by, v)}\nn={n:,}" for v, n in zip(rates[by], rates["n"], strict=True)]
    bars = ax.bar(labels, rates["pass_rate"].fillna(0), color=config.PASS_COLOR)
    ax.bar_label(
        bars,
        labels=[f"{r:.1%}" if pd.notna(r) else "—" for r in rates["pass_rate"]],
        padding=3,
        color=config.INK_PRIMARY,
    )
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(axis_title(by))
    ax.set_ylabel("Pass rate")


def _static_mean(ax, means: pd.DataFrame, by: str) -> None:
    labels = [f"{value_label(by, v)}\nn={n:,}" for v, n in zip(means[by], means["n"], strict=True)]
    bars = ax.bar(
        labels,
        means["mean"].fillna(0),
        color=config.PASS_COLOR,
        yerr=means["sem"].fillna(0),
        ecolor=config.INK_MUTED,
        capsize=3,
    )
    ax.bar_label(
        bars,
        labels=[f"{m:.1f}" if pd.notna(m) else "—" for m in means["mean"]],
        padding=6,
        color=config.INK_PRIMARY,
    )
    ax.set_ylim(0, config.GRADE_MAX)
    ax.set_xlabel(axis_title(by))
    ax.set_ylabel("Mean final grade (G3)")


def _static_box(ax, df: pd.DataFrame, by: str) -> None:
    levels = sorted(df[by].dropna().unique().tolist())
    data = [df.loc[df[by] == lv, config.TARGET_SOURCE] for lv in levels]
    bp = ax.boxplot(
        data,
        tick_labels=[
            f"{value_label(by, lv)}\nn={len(d):,}" for lv, d in zip(levels, data, strict=True)
        ],
        showmeans=True,
        patch_artist=True,
    )
    for i, box in enumerate(bp["boxes"]):
        color = config.SERIES_SEQUENCE[i % len(config.SERIES_SEQUENCE)]
        box.set(facecolor="none", edgecolor=color, linewidth=1.5)
    ax.set_xlabel(axis_title(by))
    ax.set_ylabel(axis_title("G3"))


def export_static_figures(
    df: pd.DataFrame, out_dir: Path | None = None, dpi: int = config.FIGURE_EXPORT_DPI
) -> list[Path]:
    """Render every exploratory chart as a PNG (Matplotlib, ``dpi`` resolution).

    Args:
        df: Dataset to plot (typically already filtered).
        out_dir: Destination folder; defaults to ``config.FIGURES_DIR``.
        dpi: Output resolution.

    Returns:
        Paths of the written files, in :data:`FIGURE_CATALOG` order.
    """
    plt = _mpl()
    out_dir = out_dir or config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    grades = np.arange(config.GRADE_MIN, config.GRADE_MAX + 1)

    from matplotlib.ticker import MaxNLocator

    def hist(ax):
        counts = df[config.TARGET_SOURCE].value_counts().reindex(grades, fill_value=0)
        fail = grades < config.PASS_THRESHOLD
        ax.bar(
            grades[fail],
            counts[fail],
            color=config.FAIL_COLOR,
            hatch="//",
            edgecolor=config.SURFACE,
            label=config.FAIL_LABEL,
        )
        ax.bar(grades[~fail], counts[~fail], color=config.PASS_COLOR, label=config.PASS_LABEL)
        ax.axvline(config.PASS_THRESHOLD - 0.5, color=config.INK_MUTED, ls=":", lw=1)
        ax.set_xticks(grades)
        ax.set_xlabel(axis_title("G3"))
        ax.set_ylabel("Number of students")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

    def scatter(ax):
        rng = np.random.default_rng(config.RANDOM_STATE)
        for outcome, flag, marker in ((config.PASS_LABEL, 1, "o"), (config.FAIL_LABEL, 0, "x")):
            part = df[df[config.PASS_COL] == flag]
            ax.scatter(
                part["G2"] + rng.uniform(-0.25, 0.25, len(part)),
                part["G3"] + rng.uniform(-0.25, 0.25, len(part)),
                s=14,
                alpha=0.6,
                marker=marker,
                color=config.OUTCOME_COLORS[outcome],
                label=outcome,
            )
        ax.axhline(config.PASS_THRESHOLD - 0.5, color=config.INK_MUTED, ls=":", lw=1)
        ax.set_xlabel(axis_title("G2"))
        ax.set_ylabel(axis_title("G3"))
        ax.legend()

    def corr_bar(ax):
        corr = analytics.correlations_with(df)
        neg, pos = config.DIVERGING_SCALE[0][1], config.DIVERGING_SCALE[-1][1]
        ax.barh(corr.index, corr.to_numpy(), color=[pos if v >= 0 else neg for v in corr])
        for y, v in enumerate(corr):
            ax.text(
                v + (0.02 if v >= 0 else -0.02),
                y,
                f"{v:+.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
                color=config.INK_PRIMARY,
            )
        ax.set_xlim(-1, 1)
        ax.axvline(0, color=config.AXIS_LINE, lw=1)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Pearson correlation with G3")

    def heatmap(ax):
        from matplotlib.colors import LinearSegmentedColormap

        m = analytics.correlation_matrix(df)
        cmap = LinearSegmentedColormap.from_list("div", list(config.DIVERGING_SCALE))
        im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(m)), m.columns, rotation=90)
        ax.set_yticks(range(len(m)), m.index)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(m.to_numpy()):
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=config.INK_PRIMARY
            )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")

    def pass_donut(ax):
        n_pass = int(df[config.PASS_COL].sum())
        wedges, _ = ax.pie(
            [n_pass, len(df) - n_pass],
            colors=[config.PASS_COLOR, config.FAIL_COLOR],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.38, "edgecolor": config.SURFACE, "linewidth": 2},
        )
        wedges[1].set_hatch("//")
        share = n_pass / len(df) if len(df) else 0
        ax.text(0, 0, f"{len(df):,}\nstudents", ha="center", va="center", fontsize=12)
        ax.legend(
            wedges,
            [f"Pass {share:.1%}", f"Fail {1 - share:.1%}"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
        )
        ax.set_aspect("equal")
        ax.grid(False)

    builders: dict[str, tuple[Callable, tuple[float, float]]] = {
        "01_g3_distribution.png": (hist, (9, 4.5)),
        "02_pass_rate_by_subject.png": (
            lambda ax: _static_rate(
                ax, analytics.pass_rate_by(df, config.SUBJECT_COL), config.SUBJECT_COL
            ),
            (6, 4.5),
        ),
        "03_pass_rate_by_school.png": (
            lambda ax: _static_rate(ax, analytics.pass_rate_by(df, "school"), "school"),
            (6, 4.5),
        ),
        "04_mean_g3_by_studytime.png": (
            lambda ax: _static_mean(ax, analytics.mean_by(df, "studytime"), "studytime"),
            (7, 4.5),
        ),
        "05_mean_g3_by_failures.png": (
            lambda ax: _static_mean(ax, analytics.mean_by(df, "failures"), "failures"),
            (7, 4.5),
        ),
        "06_pass_rate_by_absence_band.png": (
            lambda ax: _static_rate(
                ax, analytics.pass_rate_by(df, config.ABSENCE_BAND_COL), config.ABSENCE_BAND_COL
            ),
            (8, 4.5),
        ),
        "07_g3_box_by_subject.png": (lambda ax: _static_box(ax, df, config.SUBJECT_COL), (6, 4.5)),
        "08_g3_box_by_higher.png": (lambda ax: _static_box(ax, df, "higher"), (6, 4.5)),
        "09_g2_vs_g3_scatter.png": (scatter, (6.5, 6)),
        "10_correlation_with_g3.png": (corr_bar, (7, 7)),
        "11_correlation_heatmap.png": (heatmap, (10, 9)),
        "12_pass_fail_donut.png": (pass_donut, (5.5, 5.5)),
    }
    written: list[Path] = []
    for name, (draw, size) in builders.items():
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        ax.set_title(FIGURE_CATALOG[name], loc="left", color=config.INK_PRIMARY, fontsize=11)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Circle (donut) charts
# ---------------------------------------------------------------------------
def donut(
    labels: list[str],
    values: list[float],
    colors: list[str],
    center_title: str,
    center_value: str,
    patterns: list[str] | None = None,
    height: int = 320,
) -> go.Figure:
    """Donut chart with labelled slices (name + share) and a centre total.

    Slices carry text labels, and optional hatch patterns, so meaning never relies on colour.
    """
    total = sum(values) or 1
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            sort=False,
            direction="clockwise",
            marker={
                "colors": colors,
                "line": {"color": config.SURFACE, "width": 2},
                "pattern": {
                    "shape": patterns or [""] * len(labels),
                    "fgcolor": config.SURFACE,
                    "size": 6,
                    "solidity": 0.25,
                    "fillmode": "overlay",
                },
            },
            text=[
                f"{lab}<br>{v / total:.1%}" if v else ""
                for lab, v in zip(labels, values, strict=True)
            ],
            texttemplate="%{text}",
            textposition="outside",
            textfont={"color": config.INK_PRIMARY, "size": 12},
            hovertemplate="%{label}: %{value:,} (%{percent:.1%})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<span style='font-size:22px;color:{config.INK_PRIMARY}'><b>{center_value}</b></span>"
        f"<br><span style='font-size:12px;color:{config.INK_SECONDARY}'>{center_title}</span>",
        showarrow=False,
        x=0.5,
        y=0.5,
    )
    fig.update_layout(showlegend=False)
    fig = _style(fig, height=height)
    fig.update_layout(margin={"l": 40, "r": 40, "t": 40, "b": 40})
    return fig


def outcome_donut(df: pd.DataFrame) -> go.Figure:
    """Pass vs fail share as a donut (fail slice hatched)."""
    n_pass = int(df[config.PASS_COL].sum())
    n_fail = len(df) - n_pass
    return donut(
        [config.PASS_LABEL, config.FAIL_LABEL],
        [n_pass, n_fail],
        [config.PASS_COLOR, config.FAIL_COLOR],
        "students",
        f"{len(df):,}",
        patterns=[
            config.OUTCOME_PATTERNS[config.PASS_LABEL],
            config.OUTCOME_PATTERNS[config.FAIL_LABEL],
        ],
    )


def subject_donut(df: pd.DataFrame) -> go.Figure:
    """Share of records per subject as a donut."""
    counts = df[config.SUBJECT_COL].value_counts().sort_index()
    return donut(
        [value_label(config.SUBJECT_COL, s) for s in counts.index],
        counts.tolist(),
        [config.SERIES_SEQUENCE[i % len(config.SERIES_SEQUENCE)] for i in range(len(counts))],
        "records",
        f"{int(counts.sum()):,}",
    )


# ---------------------------------------------------------------------------
# Model evaluation figures
# ---------------------------------------------------------------------------
def metric_comparison_bar(table: pd.DataFrame, metric: str, label: str) -> go.Figure:
    """Grouped bars of one metric (± std) per model, setting A vs B side by side."""
    fig = go.Figure()
    for i, setting in enumerate(config.FEATURE_SETTINGS):
        part = table[table["setting"] == setting]
        fig.add_bar(
            x=part["model"],
            y=part[metric],
            name=f"Setting {setting}: {config.FEATURE_SETTINGS[setting]}",
            marker={"color": config.SERIES_SEQUENCE[i]},
            error_y={
                "type": "data",
                "array": part[f"{metric}_std"],
                "color": config.INK_MUTED,
                "thickness": 1,
                "width": 4,
            },
            text=[f"{v:.2f}" for v in part[metric]],
            textposition="inside",
            insidetextanchor="start",
            textfont={"color": "#ffffff"},
            hovertemplate="%{x}<br>" + label + " %{y:.3f}<extra>Setting " + setting + "</extra>",
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.05], tickformat=".1f")
    return _style(fig, "", label, 360)


def confusion_heatmap(cm: np.ndarray) -> go.Figure:
    """Confusion matrix (rows actual, cols predicted; order Fail, Pass) with counts and %."""
    labels = [config.FAIL_LABEL, config.PASS_LABEL]
    total = cm.sum() or 1
    text = [[f"<b>{v:,}</b><br>{v / total:.1%}" for v in row] for row in cm]
    seq = config.SEQUENTIAL_BLUE
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {lab}" for lab in labels],
            y=[f"Actual {lab}" for lab in labels],
            colorscale=[[0, seq[0]], [1, seq[6]]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14},
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="%{y}, %{x}: %{z:,}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig = _style(fig, "", "", 340)
    fig.update_layout(margin={"l": 100, "r": 16, "t": 40, "b": 16})
    return fig


def roc_chart(curves: dict[str, pd.DataFrame], aucs: dict[str, float]) -> go.Figure:
    """ROC curves for detecting fail, one line per model, plus the chance diagonal."""
    fig = go.Figure()
    for i, (name, pts) in enumerate(curves.items()):
        fig.add_scatter(
            x=pts["fpr"],
            y=pts["tpr"],
            mode="lines",
            name=f"{name} (AUC {aucs[name]:.2f})",
            line={"color": config.SERIES_SEQUENCE[i % len(config.SERIES_SEQUENCE)], "width": 2},
            hovertemplate="False-alarm rate %{x:.2f}<br>Fail recall %{y:.2f}<extra>"
            + name
            + "</extra>",
        )
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Chance (AUC 0.50)",
        line={"color": config.INK_MUTED, "dash": "dot", "width": 1},
        hoverinfo="skip",
    )
    fig.update_xaxes(range=[0, 1], showgrid=True)
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, "False-alarm rate (passing students flagged)", "Fail-class recall", 380)


def threshold_tradeoff(curve: pd.DataFrame, threshold: float) -> go.Figure:
    """Fail-class recall and precision vs decision threshold, with the chosen threshold marked."""
    fig = go.Figure()
    for i, (col, name, dash) in enumerate(
        (
            ("fail_recall", "Fail-class recall", "solid"),
            ("fail_precision", "Fail-class precision", "dash"),
        )
    ):
        fig.add_scatter(
            x=curve["threshold"],
            y=curve[col],
            mode="lines",
            name=name,
            line={
                "color": config.SERIES_SEQUENCE[i % len(config.SERIES_SEQUENCE)],
                "width": 2,
                "dash": dash,
            },
            hovertemplate="Threshold %{x:.2f}: %{y:.2f}<extra>" + name + "</extra>",
        )
    fig.add_vline(
        x=threshold,
        line={"color": config.INK_SECONDARY, "width": 1},
        annotation_text=f"Chosen {threshold:.2f}",
        annotation_font={"color": config.INK_SECONDARY, "size": 11},
    )
    fig.update_xaxes(range=[0.05, 0.95], showgrid=True)
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, "Decision threshold on P(fail)", "Score", 340)


def importance_bar(values: pd.Series, title: str, top: int = 15, signed: bool = False) -> go.Figure:
    """Horizontal bar of the top features; signed values use the diverging colours."""
    data = values.head(top).iloc[::-1]
    neg, pos = config.DIVERGING_SCALE[0][1], config.DIVERGING_SCALE[-1][1]
    colors = [pos if v >= 0 else neg for v in data] if signed else config.PASS_COLOR
    fig = go.Figure(
        go.Bar(
            x=data.to_numpy(),
            y=data.index,
            orientation="h",
            marker={"color": colors},
            text=[f"{v:+.2f}" if signed else f"{v:.3f}" for v in data],
            textposition="outside",
            textfont={"color": config.INK_PRIMARY, "size": 11},
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            cliponaxis=False,
        )
    )
    if signed:
        lim = float(np.abs(data).max() * 1.25) or 1
        fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor=config.AXIS_LINE)
    fig.update_xaxes(showgrid=True)
    return _style(fig, title, "", max(320, 24 * len(data) + 90))


# ---------------------------------------------------------------------------
# Early-warning figures
# ---------------------------------------------------------------------------
RISK_PATTERNS: dict[str, str] = {"High": "x", "Medium": "/", "Low": ""}


def risk_donut(counts: dict[str, int]) -> go.Figure:
    """Students per risk tier as a donut; each slice has icon, label, share and pattern."""
    tiers = [t for t in ("High", "Medium", "Low")]
    return donut(
        [f"{config.RISK_ICONS[t]} {t}" for t in tiers],
        [counts.get(t, 0) for t in tiers],
        [config.RISK_COLORS[t] for t in tiers],
        "students scored",
        f"{sum(counts.values()):,}",
        patterns=[RISK_PATTERNS[t] for t in tiers],
    )


def risk_histogram(p_fail: np.ndarray | pd.Series, threshold: float) -> go.Figure:
    """Distribution of predicted P(fail) with the tier cut-offs and the flag threshold."""
    values = np.asarray(p_fail, dtype=float)
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(values, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    for tier, lo, hi in (
        ("Low", 0, config.RISK_MEDIUM),
        ("Medium", config.RISK_MEDIUM, config.RISK_HIGH),
        ("High", config.RISK_HIGH, 1.01),
    ):
        mask = (centers >= lo) & (centers < hi)
        fig.add_bar(
            x=centers[mask],
            y=counts[mask],
            width=0.048,
            name=f"{config.RISK_ICONS[tier]} {tier}",
            marker={
                "color": config.RISK_COLORS[tier],
                "pattern": {
                    "shape": RISK_PATTERNS[tier],
                    "fgcolor": config.SURFACE,
                    "solidity": 0.25,
                    "fillmode": "overlay",
                },
            },
            hovertemplate="P(fail) %{x:.2f} ± 0.025<br>%{y} students<extra>" + tier + "</extra>",
        )
    fig.add_vline(
        x=threshold,
        line={"color": config.INK_PRIMARY, "width": 1, "dash": "dot"},
        annotation_text=f"Flag threshold {threshold:.2f}",
        annotation_font={"color": config.INK_SECONDARY, "size": 11},
    )
    fig.update_xaxes(range=[0, 1], dtick=0.1, tickformat=".1f")
    return _style(fig, "Predicted probability of failing", "Number of students", 320)
