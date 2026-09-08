import matplotlib.pyplot as plt
import torch

# Standard Kilter LED role colors
ROLE_MAP = {
    2: {"color": "#00FF00", "label": "Start"},   # Green
    3: {"color": "#0088FF", "label": "Middle"},  # Blue
    4: {"color": "#FF00FF", "label": "Finish"},  # Magenta
    5: {"color": "#FF8800", "label": "Foot"},    # Orange
}

def plot_climb(
    holds: list[tuple[int, int, int]], 
    title: str = "Kilter Board Climb", 
    max_x: int = 36, 
    max_y: int = 39,
    save_path: str | None = None
):
    """
    Plots a climb on a simulated Kilter Board grid.
    
    Parameters:
        holds: List of (x, y, role) tuples.
        title: Title of the plot.
        max_x: Grid width boundary.
        max_y: Grid height boundary.
        save_path: Optional file path to save the image.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")

    # Plot all background board grid points as faint dots
    grid_x = [x for x in range(1, max_x) for _ in range(1, max_y)]
    grid_y = [y for _ in range(1, max_x) for y in range(1, max_y)]
    ax.scatter(grid_x, grid_y, color="#2A2A2A", s=15, zorder=1)

    # Track plotted roles for dynamic legend entries
    plotted_roles = set()

    # Plot holds with sequence indices
    for idx, (x, y, role) in enumerate(holds, 1):
        info = ROLE_MAP.get(role, {"color": "#FFFFFF", "label": f"Role {role}"})
        plotted_roles.add(role)
        
        # Outer glow / ring
        ax.scatter(x, y, color=info["color"], s=350, alpha=0.3, zorder=2)
        # Core LED hold point
        ax.scatter(x, y, color=info["color"], s=120, edgecolors="white", linewidth=1.5, zorder=3)
        # Sequence number annotation
        ax.annotate(
            str(idx),
            (x, y),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
            zorder=4,
        )

    # Custom legend
    legend_handles = []
    for r, info in ROLE_MAP.items():
        if r in plotted_roles:
            handle = ax.scatter([], [], color=info["color"], s=100, label=info["label"])
            legend_handles.append(handle)

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        facecolor="#1E1E1E",
        edgecolor="#333333",
        fontsize=10
    )
    for text in legend.get_texts():
        text.set_color("white")

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_aspect("equal")
    ax.set_xlabel("X Coordinate", color="white")
    ax.set_ylabel("Y Coordinate", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", pad=15, fontsize=14, weight="bold")
    ax.grid(True, color="#222222", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    
    plt.show()


if __name__ == "__main__":
    from generate import generate_climb
    from model import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(vocab_x=36, vocab_y=45, vocab_r=6).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    angle, grade = 45.0, 17.0
    generated_holds = generate_climb(model, angle, grade, device, temp=0.8)

    # Plot generated climb
    plot_climb(generated_holds, title=f"Generated Climb (Angle: {angle}°, Grade: {grade})")

    # Example: Plot direct dataset sample
    # dataset = torch.load("data/processed/dataset.pt")
    # sample_climb = dataset[0]["input_seq"][1:].tolist() # Skip META token
    # plot_climb(sample_climb, title="Dataset Ground Truth Climb")
