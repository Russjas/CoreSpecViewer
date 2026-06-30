
# interrogate_feature.py
"""Single-feature interrogation: downhole position log + position-vs-strength funnel."""
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDialogButtonBox
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from ..config import config  # for feature_detection_threshold


def list_features(hole):
    """Names with both a POS and DEP profile product."""
    keys = set(hole.product_datasets.keys())
    feats = []
    for k in keys:
        if k.endswith("POS"):
            name = k[:-len("POS")]
            if f"{name}DEP" in keys:
                feats.append(name)
    return sorted(feats)


class FeatureSelectDialog(QDialog):
    """Pick one feature to interrogate."""
    def __init__(self, features, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interrogate feature")
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Feature:", self))
        self.combo = QComboBox(self)
        self.combo.addItems(features)
        row.addWidget(self.combo)
        layout.addLayout(row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @classmethod
    def get_feature(cls, hole, parent=None):
        feats = list_features(hole)
        if not feats:
            return None
        dlg = cls(feats, parent)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.combo.currentText()


class FeatureInterrogationWindow(QDialog):
    """Pop-out with the two single-feature diagnostic plots."""
    def __init__(self, hole, name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Interrogate: {name}")
        self.resize(1100, 750)

        depths, pos, _      = hole.step_product_dataset(f"{name}POS")
        depths2, strength, _ = hole.step_product_dataset(f"{name}DEP")
        # POS and DEP share the grid; guard anyway
        if not np.allclose(depths, depths2, equal_nan=True):
            raise ValueError("POS and DEP depth grids are misaligned")

        valid = ~np.isnan(pos) & ~np.isnan(strength)
        d, p, s = np.asarray(depths)[valid], np.asarray(pos)[valid], np.asarray(strength)[valid]
        thresh = getattr(config, "feature_detection_threshold", 0.7)

        fig = Figure(figsize=(11, 7))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Panel 1 — position vs depth, coloured by strength (downhole log)
        sc1 = ax1.scatter(p, d, c=s, s=4, cmap="viridis",
                          vmin=0, vmax=1, linewidths=0)
        ax1.invert_yaxis()
        ax1.set_xlabel(f"{name} position")
        ax1.set_ylabel("Depth (m)")
        ax1.set_title("Position vs depth\n(colour = band strength)")
        ax1.grid(True, alpha=0.2)
        fig.colorbar(sc1, ax=ax1, label="strength", fraction=0.046, pad=0.04)

        # Panel 2 — position vs strength funnel, coloured by depth (cross-plot)
        sc2 = ax2.scatter(p, s, c=d, s=4, cmap="viridis", linewidths=0)
        ax2.axhline(thresh, ls="--", c="k", lw=0.8,
                    label=f"detection threshold ({thresh})")
        ax2.set_xlabel(f"{name} position")
        ax2.set_ylabel("band strength (0–1)")
        ax2.set_title("Position vs strength\n(colour = depth)")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc="lower right", fontsize=8)
        fig.colorbar(sc2, ax=ax2, label="Depth (m)", fraction=0.046, pad=0.04)

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar(canvas, self))
        layout.addWidget(canvas)