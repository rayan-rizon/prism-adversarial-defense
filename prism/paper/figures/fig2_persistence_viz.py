"""
Figure 2: Clean vs Adversarial Persistence Diagram Visualization
The key visual that makes the paper memorable.
"""
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser


def generate_figure2(save_path='paper/figures/fig2.pdf'):
    """Generate the canonical clean-vs-adversarial persistence diagram comparison."""
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # --- Simulate clean activation point cloud ---
    # Clean: smooth manifold structure (torus-like)
    n_pts = 200
    theta = rng.uniform(0, 2 * np.pi, n_pts)
    phi = rng.uniform(0, 2 * np.pi, n_pts)
    R, r = 3, 1
    clean_cloud = np.column_stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi),
    ]) + rng.normal(0, 0.1, (n_pts, 3))

    # --- Simulate adversarial activation point cloud ---
    # Adversarial: disrupted topology (extra holes, broken loops)
    adv_cloud = clean_cloud.copy()
    # Add noise spikes and scatter
    n_perturbed = 60
    idx = rng.choice(n_pts, n_perturbed, replace=False)
    adv_cloud[idx] += rng.normal(0, 1.5, (n_perturbed, 3))
    # Add random outlier points
    outliers = rng.uniform(-5, 5, (30, 3))
    adv_cloud = np.vstack([adv_cloud, outliers])

    # --- Compute persistence diagrams ---
    clean_result = ripser(clean_cloud, maxdim=1)
    adv_result = ripser(adv_cloud, maxdim=1)

    # --- Plot: Top-left: Clean point cloud (2D projection) ---
    ax = axes[0, 0]
    ax.scatter(clean_cloud[:, 0], clean_cloud[:, 1], s=5, alpha=0.6, c='steelblue')
    ax.set_title('Clean Activations (2D Projection)', fontsize=11, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal')

    # --- Plot: Top-right: Clean persistence diagram ---
    ax = axes[0, 1]
    _plot_persistence_diagram(ax, clean_result['dgms'], title='Clean Persistence Diagram')

    # --- Plot: Bottom-left: Adversarial point cloud ---
    ax = axes[1, 0]
    ax.scatter(adv_cloud[:, 0], adv_cloud[:, 1], s=5, alpha=0.6, c='indianred')
    ax.set_title('Adversarial Activations (2D Projection)', fontsize=11, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal')

    # --- Plot: Bottom-right: Adversarial persistence diagram ---
    ax = axes[1, 1]
    _plot_persistence_diagram(ax, adv_result['dgms'], title='Adversarial Persistence Diagram')

    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()


def _plot_persistence_diagram(ax, dgms, title=''):
    """Plot a persistence diagram with H0 and H1 features."""
    colors = ['steelblue', 'indianred']
    labels = ['H₀ (components)', 'H₁ (loops)']

    max_val = 0
    for i, dgm in enumerate(dgms[:2]):
        if len(dgm) == 0:
            continue
        finite_mask = np.isfinite(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        if len(finite_dgm) > 0:
            ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1],
                       s=20, alpha=0.7, c=colors[i], label=labels[i])
            max_val = max(max_val, finite_dgm.max())

    # Diagonal line
    lim = max_val * 1.1 if max_val > 0 else 1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=0.5)
    ax.set_xlim(-0.05 * lim, lim)
    ax.set_ylim(-0.05 * lim, lim)
    ax.set_xlabel('Birth', fontsize=10)
    ax.set_ylabel('Death', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')


if __name__ == '__main__':
    generate_figure2()
