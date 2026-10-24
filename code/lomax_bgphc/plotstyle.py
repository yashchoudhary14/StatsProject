"""Shared matplotlib style (colour-blind-safe categorical palette, thin marks,
recessive grid; line styles/markers give a second, print-safe encoding)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FACILITY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]        # periods / facilities
FACILITY_STYLES = ["-", "--", "-."]
FACILITY_MARKERS = ["o", "s", "^"]
METHOD_COLORS = {"MLE": "#2a78d6", "PIV": "#eb6834", "OR_MLE": "#4a3aa7",
                 "BU": "#1baf7a", "BOR": "#e87ba4"}
METHOD_STYLES = {"MLE": "-", "PIV": "--", "OR_MLE": ":", "BU": "-.", "BOR": (0, (5, 1, 1, 1))}
METHOD_MARKERS = {"MLE": "o", "PIV": "s", "OR_MLE": "D", "BU": "^", "BOR": "v"}
METHOD_LABELS = {"MLE": "MLE", "PIV": "Pivotal", "OR_MLE": "Order-restricted MLE",
                 "BU": "Bayes (unrestricted)", "BOR": "Bayes (order-restricted)"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
    })
