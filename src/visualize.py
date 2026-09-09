import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch

# Standard Kilter LED role colors
ROLE_MAP = {
    2: {"color": "#00FF00", "label": "Start"},   # Green
    3: {"color": "#0088FF", "label": "Middle"},  # Blue
    4: {"color": "#FF00FF", "label": "Finish"},  # Magenta
    5: {"color": "#FF8800", "label": "Foot"},    # Orange
}

def plot_climb_comparison(
    holds: list[tuple[int, int, int]], 
    image_path: str = "/home/tudor/Code/DynoNet/data/raw/kilter_setting.jpeg",
    title: str = "Kilter Board Climb", 
    max_x: int = 36, 
    max_y: int = 39,
    # Image calibration parameters
    grid_left: float = -5.5,
    grid_right: float = 30.5,
    grid_bottom: float = 0.0,
    grid_top: float = 20.0,
    save_path: str | None = None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    bg_color = "#121212"
    fig.patch.set_facecolor(bg_color)

    # -------------------------------------------------------------
    # LEFT PANEL: Synthetic Vector Grid Plot
    # -------------------------------------------------------------
    ax1.set_facecolor(bg_color)
    grid_x = [x for x in range(1, max_x) for _ in range(1, max_y)]
    grid_y = [y for _ in range(1, max_x) for y in range(1, max_y)]
    ax1.scatter(grid_x, grid_y, color="#2A2A2A", s=15, zorder=1)

    plotted_roles = set()

    for idx, (x, y, role) in enumerate(holds, 1):
        info = ROLE_MAP.get(role, {"color": "#FFFFFF", "label": f"Role {role}"})
        plotted_roles.add(role)
        
        ax1.scatter(x, y, color=info["color"], s=350, alpha=0.3, zorder=2)
        ax1.scatter(x, y, color=info["color"], s=120, edgecolors="white", linewidth=1.5, zorder=3)
        ax1.annotate(
            str(idx),
            (x, y),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
            zorder=4,
        )

    ax1.set_xlim(0, max_x)
    ax1.set_ylim(0, max_y)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X Coordinate", color="white")
    ax1.set_ylabel("Y Coordinate", color="white")
    ax1.tick_params(colors="white")
    ax1.set_title("1. Synthetic Grid View", color="white", pad=15, fontsize=14, weight="bold")
    ax1.grid(True, color="#222222", linestyle="--", linewidth=0.5)

    # -------------------------------------------------------------
    # RIGHT PANEL: Real Photo Grid Overlay
    # -------------------------------------------------------------
    ax2.set_facecolor(bg_color)
    
    img = mpimg.imread(image_path)
    extent = [grid_left, grid_right, grid_bottom, grid_top]
    
    # Changed origin to 'upper' so the image renders right-side up
    ax2.imshow(img, extent=extent, origin="upper", aspect="equal")

    for idx, (x, y, role) in enumerate(holds, 1):
        info = ROLE_MAP.get(role, {"color": "#FFFFFF"})
        
        ax2.scatter(x, y, color=info["color"], s=450, alpha=0.35, edgecolors="none", zorder=2)
        ax2.scatter(x, y, facecolors="none", edgecolors=info["color"], s=150, linewidth=2.0, zorder=3)
        ax2.annotate(
            str(idx),
            (x + 0.5, y + 0.5),
            color="white",
            fontsize=9,
            weight="bold",
            zorder=4,
        )

    ax2.set_xlim(0, max_x)
    ax2.set_ylim(0, max_y)
    ax2.set_xlabel("X Coordinate", color="white")
    ax2.set_ylabel("Y Coordinate", color="white")
    ax2.tick_params(colors="white")
    ax2.set_title("2. Physical Board Overlay", color="white", pad=15, fontsize=14, weight="bold")

    # Shared Legend
    legend_handles = []
    for r, info in ROLE_MAP.items():
        if r in plotted_roles:
            handle = ax1.scatter([], [], color=info["color"], s=100, label=info["label"])
            legend_handles.append(handle)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_handles),
            facecolor="#1E1E1E",
            edgecolor="#333333",
            fontsize=12,
            labelcolor="white",
        )

    fig.suptitle(title, color="white", fontsize=16, weight="bold", y=1.06)
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

    angle, grade = 45.0, 13.0
    generated_holds = generate_climb(model, angle, grade, device, temp=0.8)

    plot_climb_comparison(
        generated_holds,
        image_path="/home/tudor/Code/DynoNet/data/raw/kilter_setting.jpeg",
        title=f"Generated Climb (Angle: {angle}°, Grade: {grade})",
        max_x=36,
        max_y=39,
        grid_left=0.0,
        grid_right=36.0,
        grid_bottom=0.0,
        grid_top=40.0
    )
