"""
Scaling Laws for Neural Language Models — Extended Analysis
Kaplan et al. (2020) — arXiv:2001.08361

Demonstrates the more advanced results from the paper:
  1. Joint loss L(N,D) overfitting surface
  2. Training curves L(N,S) for multiple model sizes
  3. Critical batch size B_crit(L)
  4. Overfitting condition D >= 5000 * N^0.74

Output: outputs/scaling_laws_extended.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# CONSTANTS (Kaplan et al., Table 5 / Section 1.2)
# ============================================================

alpha_N = 0.076
alpha_D = 0.095
alpha_C = 0.050
alpha_S = 0.76
alpha_B = 0.21

Nc = 8.8e13
Dc = 5.4e13
Cc = 3.1e8
Sc = 2.1e3
B_star = 2.0e8

# ============================================================
# SCALING-LAW FUNCTIONS
# ============================================================


def loss_joint(N, D):
    """
    Joint loss L(N, D) — Equation 1.5.

    Models the interaction between model size and dataset size.
    When D is small relative to N, the model overfits: loss rises above
    the convergence floor L(N). This surface is the basis for the
    overfitting condition D >= 5000 * N^0.74.

    Why it matters: this equation lets practitioners predict whether a
    given (N, D) pair will waste capacity through overfitting, and
    quantifies how much additional data is needed when scaling up N.
    """
    return ((Nc / N) ** (alpha_N / alpha_D) + (Dc / D)) ** alpha_D


def loss_training_curve(N, S):
    """
    Training curve L(N, S) — Equation 1.6.

    Decomposes total loss into an irreducible model-size term plus a
    training-progress term that decays as a power law in steps S.

    Why it matters: this equation shows that early stopping is optimal —
    at some point, additional steps yield negligible loss reduction,
    and the compute is better spent on a larger model.
    """
    return (Nc / N) ** alpha_N + (Sc / np.maximum(S, 1)) ** alpha_S


def critical_batch_size(L):
    """
    Critical batch size B_crit(L) — Equation 1.4.

    The batch size at which the gradient noise scale equals 1, giving
    roughly equal contributions from gradient noise and curvature.
    Below B_crit, larger batches help; above it, diminishing returns.

    Key property: B_crit depends only on the current loss L, not on
    model size N — a surprising universality result.

    Why it matters: this tells practitioners how to set batch size
    efficiently as training progresses and loss decreases.
    """
    return B_star / (L ** (1.0 / alpha_B))


def min_data_for_model(N):
    """
    Overfitting condition: D >= 5000 * N^0.74 — Section 4.

    The minimum dataset size to avoid significant overfitting for a
    model of size N. The sublinear exponent (0.74 < 1) means that
    data requirements grow more slowly than model size — a favorable
    scaling property.

    Why it matters: this gives a concrete lower bound on dataset size
    when planning a training run at a given model scale.
    """
    return 5000.0 * N ** 0.74


# ============================================================
# VISUALIZATION — 4 panels
# ============================================================

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

style = dict(
    facecolor='#161b22',
    labelcolor='#e6edf3',
    tickcolor='#8b949e',
    gridcolor='#21262d',
)


def style_ax(ax, title, xlabel, ylabel):
    """Apply consistent dark-theme styling to an axes object."""
    ax.set_facecolor(style['facecolor'])
    ax.tick_params(colors=style['tickcolor'], labelsize=9)
    ax.xaxis.label.set_color(style['labelcolor'])
    ax.yaxis.label.set_color(style['labelcolor'])
    ax.title.set_color('#58a6ff')
    ax.set_title(title, fontsize=11, pad=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=style['gridcolor'], alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')


# ------------------------------------------------------------------
# Panel 1: Joint loss L(N,D) — overfitting surface
# ------------------------------------------------------------------
# Shows how loss depends on both model size and dataset size.
# Contour lines are iso-loss curves; the overfitting boundary
# (D = 5000 * N^0.74) separates the "safe" region from the
# region where insufficient data degrades performance.
# ------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
N_grid = np.logspace(6, 11, 250)
D_grid = np.logspace(7, 13, 250)
NN, DD = np.meshgrid(N_grid, D_grid)
LL = loss_joint(NN, DD)

levels = np.linspace(1.4, 5.0, 16)
cf = ax1.contourf(np.log10(NN), np.log10(DD), LL,
                  levels=levels, cmap='viridis', alpha=0.9)
ax1.contour(np.log10(NN), np.log10(DD), LL,
            levels=levels, colors='white', linewidths=0.3, alpha=0.4)
cbar = fig.colorbar(cf, ax=ax1, pad=0.02)
cbar.set_label('Loss L(N,D)', color=style['labelcolor'], fontsize=8)
cbar.ax.tick_params(colors=style['tickcolor'])

# Overfitting boundary
N_line = np.logspace(6, 11, 200)
D_line = min_data_for_model(N_line)
ax1.plot(np.log10(N_line), np.log10(D_line),
         color='#f78166', linewidth=2.5, linestyle='--',
         label='D = 5000 · N^0.74')
ax1.fill_between(np.log10(N_line), np.log10(D_line), 7,
                 color='#f78166', alpha=0.08)
ax1.text(9.5, 8.0, 'OVERFITTING\nREGION', color='#f78166',
         fontsize=9, fontweight='bold', ha='center', alpha=0.8)

style_ax(ax1, 'Joint Loss Surface — L(N, D)',
         'log₁₀(Parameters N)', 'log₁₀(Tokens D)')
ax1.legend(fontsize=8, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d', loc='upper left')


# ------------------------------------------------------------------
# Panel 2: Training curves L(N,S) for multiple model sizes
# ------------------------------------------------------------------
# Each curve shows how loss decreases with training steps for a
# fixed model size. Larger models start at a lower loss floor
# (the (Nc/N)^alpha_N term) and converge faster.  The gap between
# curves quantifies the benefit of scaling N vs. training longer.
# ------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
S_range = np.logspace(1, 6, 400)
model_sizes = [1e7, 1e8, 5e8, 1e9, 1e10]
colors_tc = ['#f78166', '#ffa657', '#d2a8ff', '#58a6ff', '#3fb950']

for N_val, color in zip(model_sizes, colors_tc):
    L_curve = loss_training_curve(N_val, S_range)
    label = f'N = {N_val:.0e}'
    ax2.semilogx(S_range, L_curve, color=color, linewidth=2, label=label)
    # Mark the approximate convergence point (where step term < 1% of total)
    L_floor = (Nc / N_val) ** alpha_N
    converge_idx = np.argmax(L_curve < L_floor * 1.01)
    if converge_idx > 0:
        ax2.scatter(S_range[converge_idx], L_curve[converge_idx],
                    color=color, s=40, zorder=5, edgecolors='white',
                    linewidths=0.5)

style_ax(ax2, 'Training Curves — L(N, S)',
         'Training Steps (S)', 'Test Loss')
ax2.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d', loc='upper right')
ax2.set_ylim(1.0, 8.0)
ax2.annotate('Larger models reach\nlower loss floors',
             xy=(1e5, 2.0), fontsize=8, color='#8b949e',
             fontstyle='italic')


# ------------------------------------------------------------------
# Panel 3: Critical batch size B_crit(L)
# ------------------------------------------------------------------
# B_crit depends only on the current loss — not on model size.
# As training progresses and loss decreases, B_crit grows,
# meaning larger batches become useful later in training.
# This guides dynamic batch-size schedules.
# ------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
L_range = np.linspace(1.5, 5.0, 300)
B_crit_vals = critical_batch_size(L_range)

ax3.semilogy(L_range, B_crit_vals, color='#d2a8ff', linewidth=2.5,
             label='B_crit(L) = B* / L^(1/α_B)')
ax3.axhline(y=B_star, color='#ffa657', ls='--', alpha=0.5, lw=1)
ax3.text(4.5, B_star * 1.3, f'B* = {B_star:.0e}', color='#ffa657',
         fontsize=8)

# Shade the efficient region
ax3.fill_between(L_range, B_crit_vals, 1e4,
                 color='#3fb950', alpha=0.06)
ax3.text(3.5, 3e5, 'Efficient region\n(B < B_crit)', color='#3fb950',
         fontsize=8, ha='center', fontstyle='italic')
ax3.fill_between(L_range, B_crit_vals, 1e12,
                 color='#f78166', alpha=0.06)
ax3.text(3.5, 5e9, 'Diminishing returns\n(B > B_crit)', color='#f78166',
         fontsize=8, ha='center', fontstyle='italic')

style_ax(ax3, 'Critical Batch Size — B_crit(L)',
         'Loss L', 'Critical Batch Size (tokens)')
ax3.legend(fontsize=8, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d', loc='upper right')
ax3.set_ylim(1e4, 1e12)


# ------------------------------------------------------------------
# Panel 4: Overfitting condition — D >= 5000 * N^0.74
# ------------------------------------------------------------------
# Plots the minimum dataset size as a function of model size.
# The sublinear exponent (0.74) is favorable: doubling N requires
# only ~1.67x more data, not 2x.  Points above the line are safe;
# points below risk overfitting.
# ------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
N_range_of = np.logspace(6, 12, 300)
D_min = min_data_for_model(N_range_of)

ax4.loglog(N_range_of, D_min, color='#f78166', linewidth=2.5,
           label='D_min = 5000 · N^0.74')
# Reference: linear scaling D = N for comparison
ax4.loglog(N_range_of, N_range_of, color='#8b949e', linewidth=1,
           linestyle=':', label='D = N (linear)')
ax4.fill_between(N_range_of, D_min, 1e15,
                 color='#3fb950', alpha=0.06)
ax4.fill_between(N_range_of, D_min, 1,
                 color='#f78166', alpha=0.06)

# Mark well-known models
models = {
    'GPT-2': (1.5e9, 40e9),
    'GPT-3': (175e9, 300e9),
    'Chinchilla': (70e9, 1.4e12),
}
marker_colors = {'GPT-2': '#58a6ff', 'GPT-3': '#ffa657',
                 'Chinchilla': '#3fb950'}
for name, (n, d) in models.items():
    ax4.scatter(n, d, color=marker_colors[name], s=80, zorder=5,
                edgecolors='white', linewidths=0.8)
    ax4.annotate(name, (n, d), textcoords="offset points",
                 xytext=(8, 8), fontsize=8, color=marker_colors[name],
                 fontweight='bold')

ax4.text(1e10, 5e4, 'OVERFITTING\nRISK', color='#f78166',
         fontsize=10, fontweight='bold', ha='center', alpha=0.7)
ax4.text(1e8, 5e12, 'SAFE REGION', color='#3fb950',
         fontsize=10, fontweight='bold', ha='center', alpha=0.7)

style_ax(ax4, 'Overfitting Condition — D ≥ 5000 · N^0.74',
         'Parameters (N)', 'Minimum Dataset Size (tokens)')
ax4.legend(fontsize=8, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d', loc='upper left')
ax4.set_ylim(1e4, 1e15)


# --- Suptitle ---
fig.suptitle(
    'Scaling Laws — Extended Analysis: Overfitting, Training Curves, and Batch Size\n'
    'Kaplan et al. (2020) — arXiv:2001.08361',
    color='#e6edf3', fontsize=13, fontweight='bold', y=0.99
)

# --- Save ---
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'scaling_laws_extended.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved: {out_path}")

# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 62)
print("EXTENDED ANALYSIS — NUMERICAL RESULTS")
print("=" * 62)

print("\nOverfitting Condition: D_min = 5000 * N^0.74")
print(f"{'N (params)':>15} | {'D_min (tokens)':>15} | {'Ratio D/N':>10}")
print("-" * 46)
for N in [1e7, 1e8, 1e9, 1e10, 1e11]:
    D = min_data_for_model(N)
    print(f"{N:>15.0e} | {D:>15.2e} | {D / N:>10.2f}")

print("\nCritical Batch Size at Various Loss Levels")
print(f"{'Loss':>8} | {'B_crit (tokens)':>16}")
print("-" * 28)
for L in [4.0, 3.5, 3.0, 2.5, 2.0]:
    print(f"{L:>8.1f} | {critical_batch_size(L):>16.2e}")

print("\nKey Insight: as loss decreases, B_crit grows — use larger")
print("batches later in training for maximum compute efficiency.")
