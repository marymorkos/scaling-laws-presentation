"""
Scaling Laws for Neural Language Models — Core Demonstration
Kaplan et al. (2020) — arXiv:2001.08361

Visualizes the three individual power laws (L(N), L(D), L(C_min)),
the compute-efficient frontier, and optimal model size vs. compute.
Output: outputs/scaling_laws_demo.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# CONSTANTS (fitted by Kaplan et al., Table 5 / Section 1.2)
# ============================================================

alpha_N = 0.076    # exponent — model size
alpha_D = 0.095    # exponent — dataset size
alpha_C = 0.050    # exponent — compute (optimal allocation)
alpha_S = 0.76     # exponent — training steps
alpha_B = 0.21     # exponent — batch size

Nc = 8.8e13        # parameter scale constant
Dc = 5.4e13        # token scale constant
Cc = 3.1e8         # compute scale constant (PF-days)
Sc = 2.1e3         # steps scale constant
B_star = 2.0e8     # batch size constant (tokens)

# ============================================================
# POWER LAW FUNCTIONS
# ============================================================

def loss_vs_parameters(N):
    """L(N) = (Nc / N)^alpha_N — convergence loss for model size N."""
    return (Nc / N) ** alpha_N


def loss_vs_dataset(D):
    """L(D) = (Dc / D)^alpha_D — data-limited loss for D tokens."""
    return (Dc / D) ** alpha_D


def loss_vs_compute(C_min):
    """L(Cmin) = (Cc / Cmin)^alpha_C — compute-optimal loss."""
    return (Cc / C_min) ** alpha_C


def loss_joint(N, D):
    """L(N, D) = [(Nc/N)^(alpha_N/alpha_D) + Dc/D]^alpha_D — joint loss."""
    return ((Nc / N) ** (alpha_N / alpha_D) + (Dc / D)) ** alpha_D


def optimal_model_size(C_min):
    """N_opt proportional to Cmin^0.73 — optimal model size for budget C_min."""
    Ne = 1.3e9
    return Ne * (C_min ** 0.73)


# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

style = dict(
    facecolor='#161b22',
    labelcolor='#e6edf3',
    tickcolor='#8b949e',
    gridcolor='#21262d',
)


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(style['facecolor'])
    ax.tick_params(colors=style['tickcolor'])
    ax.xaxis.label.set_color(style['labelcolor'])
    ax.yaxis.label.set_color(style['labelcolor'])
    ax.title.set_color('#58a6ff')
    ax.set_title(title, fontsize=11, pad=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=style['gridcolor'], alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')


# --- Plot 1: L(N) — Loss vs. Model Size ---
ax1 = fig.add_subplot(gs[0, 0])
N_range = np.logspace(6, 11, 300)
ax1.loglog(N_range, loss_vs_parameters(N_range),
           color='#58a6ff', linewidth=2.5, label='L(N) = (Nc/N)^0.076')
example_Ns = [1e7, 1e8, 1e9, 1e10]
ax1.scatter(example_Ns, [loss_vs_parameters(n) for n in example_Ns],
            color='#f78166', zorder=5, s=70, edgecolors='white',
            linewidths=0.5, label='Example scales')
style_ax(ax1, 'Loss vs. Model Size — L(N)', 'Parameters (N)', 'Test Loss')
ax1.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d')

# --- Plot 2: L(D) — Loss vs. Dataset Size ---
ax2 = fig.add_subplot(gs[0, 1])
D_range = np.logspace(8, 13, 300)
ax2.loglog(D_range, loss_vs_dataset(D_range),
           color='#3fb950', linewidth=2.5, label='L(D) = (Dc/D)^0.095')
style_ax(ax2, 'Loss vs. Dataset Size — L(D)', 'Tokens (D)', 'Test Loss')
ax2.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d')

# --- Plot 3: L(Cmin) — Loss vs. Compute ---
ax3 = fig.add_subplot(gs[0, 2])
C_range = np.logspace(-5, 4, 300)
ax3.loglog(C_range, loss_vs_compute(C_range),
           color='#d2a8ff', linewidth=2.5, label='L(Cmin) = (Cc/Cmin)^0.050')
style_ax(ax3, 'Loss vs. Compute — L(C_min)', 'Compute (PF-days)', 'Test Loss')
ax3.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d')

# --- Plot 4: Compute-Efficient Frontier (joint loss contours) ---
ax4 = fig.add_subplot(gs[1, 0])
N_grid = np.logspace(6, 11, 200)
D_grid = np.logspace(8, 13, 200)
NN, DD = np.meshgrid(N_grid, D_grid)
LL = loss_joint(NN, DD)
levels = np.linspace(1.5, 4.5, 12)
cf = ax4.contourf(np.log10(NN), np.log10(DD), LL, levels=levels,
                  cmap='cool', alpha=0.85)
ax4.contour(np.log10(NN), np.log10(DD), LL, levels=levels,
            colors='white', linewidths=0.3, alpha=0.4)
cbar = fig.colorbar(cf, ax=ax4, pad=0.02)
cbar.set_label('Loss L(N,D)', color=style['labelcolor'], fontsize=8)
cbar.ax.tick_params(colors=style['tickcolor'])
# Overfitting boundary: D = 5000 * N^0.74
N_boundary = np.logspace(6, 11, 200)
D_boundary = 5000 * N_boundary ** 0.74
ax4.plot(np.log10(N_boundary), np.log10(D_boundary),
         color='#f78166', linewidth=2, linestyle='--',
         label='D = 5000 · N^0.74 (overfit boundary)')
style_ax(ax4, 'Compute-Efficient Frontier — L(N, D)',
         'log₁₀(Parameters)', 'log₁₀(Tokens)')
ax4.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d', loc='lower right')

# --- Plot 5: Optimal Model Size vs. Compute ---
ax5 = fig.add_subplot(gs[1, 1])
C_opt = np.logspace(-3, 4, 300)
N_opt = optimal_model_size(C_opt)
ax5.loglog(C_opt, N_opt, color='#ffa657', linewidth=2.5,
           label='N_opt ∝ C^0.73')
# Reference lines
ax5.axhline(y=1.5e9, color='#58a6ff', ls='--', alpha=0.6, lw=1)
ax5.text(1e-2, 1.8e9, 'GPT-2 (1.5B)', color='#58a6ff', fontsize=7)
ax5.axhline(y=1.75e11, color='#3fb950', ls='--', alpha=0.6, lw=1)
ax5.text(1e-2, 2.1e11, 'GPT-3 (175B)', color='#3fb950', fontsize=7)
style_ax(ax5, 'Optimal Model Size vs. Compute',
         'Compute Budget (PF-days)', 'Optimal N')
ax5.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3',
           edgecolor='#30363d')

# --- Plot 6: Power-Law Exponents Summary ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(style['facecolor'])
for spine in ax6.spines.values():
    spine.set_edgecolor('#30363d')

factors = ['Model (α_N)', 'Data (α_D)', 'Compute (α_C)',
           'Steps (α_S)', 'Batch (α_B)']
exponents = [alpha_N, alpha_D, alpha_C, alpha_S, alpha_B]
bar_colors = ['#58a6ff', '#3fb950', '#d2a8ff', '#ffa657', '#f78166']

bars = ax6.bar(factors, exponents, color=bar_colors, alpha=0.9,
               edgecolor='#30363d', linewidth=0.8)
for bar, val in zip(bars, exponents):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val}', ha='center', va='bottom', color='#e6edf3',
             fontsize=9, fontweight='bold')
ax6.tick_params(colors=style['tickcolor'], labelsize=8)
ax6.set_title('Power-Law Exponents',
              color='#58a6ff', fontsize=11, pad=12, fontweight='bold')
ax6.set_ylabel('Exponent Value', color=style['labelcolor'], fontsize=9)
ax6.grid(True, color=style['gridcolor'], alpha=0.5, axis='y')

# --- Suptitle ---
fig.suptitle(
    'Scaling Laws for Neural Language Models — Kaplan et al. (2020)\n'
    'Performance follows power laws with Model Size (N), Data (D), and Compute (C)',
    color='#e6edf3', fontsize=13, fontweight='bold', y=0.99
)

# --- Save ---
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'scaling_laws_demo.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved: {out_path}")

# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 62)
print("SCALING LAWS — NUMERICAL DEMONSTRATION")
print("=" * 62)

print("\nL(N): Loss vs. Model Size (convergence regime)")
print(f"{'Parameters':>15} | {'Loss':>10} | {'Improvement':>12}")
print("-" * 45)
prev = None
for N in [1e6, 1e7, 1e8, 1e9, 1e10]:
    L = loss_vs_parameters(N)
    imp = f"{(prev - L) / prev * 100:.1f}%" if prev else "—"
    print(f"{N:>15.0e} | {L:>10.4f} | {imp:>12}")
    prev = L

print("\nL(D): Loss vs. Dataset Size")
print(f"{'Tokens':>15} | {'Loss':>10}")
print("-" * 30)
for D in [1e8, 1e9, 1e10, 1e11, 1e12]:
    print(f"{D:>15.0e} | {loss_vs_dataset(D):>10.4f}")

print("\nOptimal Allocation: N_opt vs. Compute Budget")
print(f"{'PF-days':>12} | {'N_opt':>12} | {'Note':>20}")
print("-" * 50)
for C in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    N_o = optimal_model_size(C)
    note = ("small run" if C < 0.1 else "GPT-2 regime"
            if C < 5 else "GPT-3 regime" if C < 500 else "frontier")
    print(f"{C:>12.2f} | {N_o:>12.2e} | {note:>20}")

print("\nKey Takeaway:")
print("  10x more compute => model grows ~5x, steps barely change.")
print("  Spend budget on BIGGER MODELS, not more training steps.")
