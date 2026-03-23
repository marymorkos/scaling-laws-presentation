"""
Scaling Laws for Neural Language Models — Code Demonstration
Kaplan et al. (2020) — arXiv:2001.08361

This script demonstrates the core power-law relationships from the paper
and visualizes the compute-efficient frontier.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# CORE SCALING LAW EQUATIONS (from paper, Section 1.2)
# ============================================================

# Power law exponents (fitted empirically by Kaplan et al.)
alpha_N = 0.076   # exponent for model size
alpha_D = 0.095   # exponent for dataset size
alpha_C = 0.050   # exponent for compute (optimal allocation)
alpha_S = 0.76    # exponent for training steps
alpha_B = 0.21    # exponent for batch size

# Scale constants (tokenization-dependent, from Table 5)
Nc = 8.8e13       # parameter scale constant
Dc = 5.4e13       # token scale constant
Cc = 3.1e8        # compute scale constant (PF-days)
Sc = 2.1e3        # steps scale constant
B_star = 2.0e8    # batch size constant (tokens)


def loss_vs_parameters(N):
    """
    L(N) = (Nc / N)^alpha_N
    Loss as a function of model size, trained to convergence on infinite data.
    Equation 1.1 from the paper.
    """
    return (Nc / N) ** alpha_N


def loss_vs_dataset(D):
    """
    L(D) = (Dc / D)^alpha_D
    Loss as a function of dataset size, large model with early stopping.
    Equation 1.2 from the paper.
    """
    return (Dc / D) ** alpha_D


def loss_vs_compute(C_min):
    """
    L(Cmin) = (Cc / Cmin)^alpha_C
    Loss as a function of optimally allocated compute.
    Equation 1.3 from the paper.
    """
    return (Cc / C_min) ** alpha_C


def loss_joint(N, D):
    """
    L(N, D) = [ (Nc/N)^(alpha_N/alpha_D) + (Dc/D) ]^alpha_D
    Joint loss equation accounting for both model size and dataset size.
    Equation 1.5 from the paper — governs overfitting behavior.
    """
    return ((Nc / N) ** (alpha_N / alpha_D) + (Dc / D)) ** alpha_D


def optimal_model_size(C_min):
    """
    N_opt ∝ Cmin^0.73
    Optimal model size for a given compute budget.
    Equation 6.1 from the paper.
    """
    Ne = 1.3e9  # scale constant from paper
    return Ne * (C_min ** 0.73)


def critical_batch_size(L):
    """
    Bcrit(L) = B* / L^(1/alpha_B)
    Critical batch size as a function of current loss.
    Equation 1.4 from the paper.
    """
    return B_star / (L ** (1 / alpha_B))


# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

plot_style = {
    'facecolor': '#161b22',
    'labelcolor': '#e6edf3',
    'tickcolor': '#8b949e',
    'gridcolor': '#21262d',
}

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(plot_style['facecolor'])
    ax.tick_params(colors=plot_style['tickcolor'])
    ax.xaxis.label.set_color(plot_style['labelcolor'])
    ax.yaxis.label.set_color(plot_style['labelcolor'])
    ax.title.set_color('#58a6ff')
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=plot_style['gridcolor'], alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# --- PLOT 1: Loss vs Model Size ---
ax1 = fig.add_subplot(gs[0, 0])
N_range = np.logspace(6, 11, 200)
L_N = loss_vs_parameters(N_range)
ax1.loglog(N_range, L_N, color='#58a6ff', linewidth=2.5, label='L(N) power law')
ax1.scatter([1e7, 1e8, 1e9, 1e10], 
            [loss_vs_parameters(n) for n in [1e7, 1e8, 1e9, 1e10]],
            color='#f78166', zorder=5, s=60, label='Example models')
style_ax(ax1, 'Loss vs Model Size\nL(N) = (Nc/N)^α_N', 
         'Parameters (N)', 'Cross-Entropy Loss')
ax1.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')

# --- PLOT 2: Loss vs Dataset Size ---
ax2 = fig.add_subplot(gs[0, 1])
D_range = np.logspace(8, 13, 200)
L_D = loss_vs_dataset(D_range)
ax2.loglog(D_range, L_D, color='#3fb950', linewidth=2.5, label='L(D) power law')
style_ax(ax2, 'Loss vs Dataset Size\nL(D) = (Dc/D)^α_D', 
         'Dataset Size (tokens)', 'Cross-Entropy Loss')
ax2.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')

# --- PLOT 3: Loss vs Compute ---
ax3 = fig.add_subplot(gs[0, 2])
C_range = np.logspace(-5, 2, 200)
L_C = loss_vs_compute(C_range)
ax3.loglog(C_range, L_C, color='#d2a8ff', linewidth=2.5, label='L(Cmin) power law')
style_ax(ax3, 'Loss vs Compute\nL(Cmin) = (Cc/Cmin)^α_C', 
         'Compute (PF-days)', 'Cross-Entropy Loss')
ax3.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')

# --- PLOT 4: Optimal Model Size vs Compute ---
ax4 = fig.add_subplot(gs[1, 0])
C_opt = np.logspace(-3, 3, 200)
N_opt = optimal_model_size(C_opt)
ax4.loglog(C_opt, N_opt, color='#ffa657', linewidth=2.5)
ax4.axvline(x=1, color='#f78166', linestyle='--', alpha=0.7, label='1 PF-day')
ax4.axhline(y=1.75e11, color='#58a6ff', linestyle='--', alpha=0.7, label='~GPT-3 size (175B)')
style_ax(ax4, 'Optimal Model Size vs Compute\nN_opt ∝ Cmin^0.73', 
         'Compute Budget (PF-days)', 'Optimal Parameters')
ax4.legend(fontsize=7, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')

# --- PLOT 5: Joint Loss — Overfitting Surface ---
ax5 = fig.add_subplot(gs[1, 1])
N_vals = np.logspace(7, 10, 6)
D_range2 = np.logspace(8, 13, 200)
colors = plt.cm.cool(np.linspace(0.2, 0.9, len(N_vals)))
for i, N_val in enumerate(N_vals):
    L_joint = loss_joint(N_val, D_range2)
    label = f'N={N_val:.0e}'
    ax5.loglog(D_range2, L_joint, color=colors[i], linewidth=1.8, label=label)
style_ax(ax5, 'Joint Loss L(N,D)\nOverfitting as D decreases', 
         'Dataset Size (tokens)', 'Cross-Entropy Loss')
ax5.legend(fontsize=6, facecolor='#21262d', labelcolor='#e6edf3', 
           edgecolor='#30363d', loc='upper right')

# --- PLOT 6: Power Law Exponents Summary ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(plot_style['facecolor'])
for spine in ax6.spines.values():
    spine.set_edgecolor('#30363d')

factors = ['Model Size\n(α_N)', 'Dataset\n(α_D)', 'Compute\n(α_C)', 
           'Steps\n(α_S)', 'Batch\n(α_B)']
exponents = [alpha_N, alpha_D, alpha_C, alpha_S, alpha_B]
bar_colors = ['#58a6ff', '#3fb950', '#d2a8ff', '#ffa657', '#f78166']

bars = ax6.bar(factors, exponents, color=bar_colors, alpha=0.85, edgecolor='#30363d')
for bar, val in zip(bars, exponents):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val}', ha='center', va='bottom', color='#e6edf3', fontsize=8, fontweight='bold')

ax6.set_facecolor(plot_style['facecolor'])
ax6.tick_params(colors=plot_style['tickcolor'])
ax6.set_title('Power Law Exponents\n(higher = faster improvement with scale)', 
              color='#58a6ff', fontsize=10, pad=10)
ax6.set_ylabel('Exponent Value', color=plot_style['labelcolor'], fontsize=8)
ax6.grid(True, color=plot_style['gridcolor'], alpha=0.5, axis='y')

# Main title
fig.suptitle('Scaling Laws for Neural Language Models — Kaplan et al. (2020)\n'
             'Performance follows power laws with Model Size (N), Data (D), and Compute (C)',
             color='#e6edf3', fontsize=12, fontweight='bold', y=0.98)

plt.savefig('/home/claude/scaling-laws-presentation/scaling_laws_demo.png', 
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Plot saved: scaling_laws_demo.png")

# ============================================================
# NUMERICAL DEMONSTRATION
# ============================================================

print("\n" + "="*60)
print("SCALING LAWS — NUMERICAL DEMONSTRATION")
print("="*60)

print("\n📊 L(N): Loss vs Model Size (trained to convergence)")
print(f"{'Parameters':>15} | {'Predicted Loss':>14} | {'% Improvement':>14}")
print("-" * 50)
prev = None
for N in [1e6, 1e7, 1e8, 1e9, 1e10]:
    L = loss_vs_parameters(N)
    improvement = f"{(prev - L)/prev*100:.1f}%" if prev else "baseline"
    print(f"{N:>15.0e} | {L:>14.4f} | {improvement:>14}")
    prev = L

print("\n📊 L(D): Loss vs Dataset Size")
print(f"{'Tokens':>15} | {'Predicted Loss':>14}")
print("-" * 35)
for D in [1e8, 1e9, 1e10, 1e11, 1e12]:
    L = loss_vs_dataset(D)
    print(f"{D:>15.0e} | {L:>14.4f}")

print("\n📊 Optimal Allocation: Where should compute go?")
print(f"{'Compute (PF-days)':>20} | {'Optimal N':>12} | {'Key Insight':>30}")
print("-" * 70)
for C in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    N_o = optimal_model_size(C)
    insight = "Small experiment" if C < 0.1 else \
              "GPT-2 scale" if C < 1 else \
              "GPT-3 scale" if C < 100 else "Future frontier"
    print(f"{C:>20.2f} | {N_o:>12.2e} | {insight:>30}")

print("\n💡 KEY TAKEAWAY:")
print("   Every 10x increase in compute → optimal model grows ~5x")
print("   Training steps barely increase (∝ Cmin^0.03)")
print("   → Spend compute on BIGGER MODELS, not more steps")
print("\n⚠️  CHINCHILLA CORRECTION (2022):")
print("   Hoffmann et al. found data should scale equally with model size")
print("   GPT-3 was likely undertrained — too big, not enough data")
print("   Modern models (Gemini, GPT-4) use more balanced data/model ratios")
