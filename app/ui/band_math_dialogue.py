"""
Dialog for creating band-maths expressions.

Collects a user-defined name and expression string, and displays syntax
guidance for the band-maths evaluation system. Evaluation and processing
are handled elsewhere.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QDialogButtonBox,
    QTextEdit, QCheckBox
)


class BandMathsDialog(QDialog):
    """
    Dialog for creating band-maths expressions.

    Collects a user-defined name and expression string, and displays syntax
    guidance for the band-maths evaluation system. Evaluation and processing
    are handled elsewhere.
    """

    def __init__(
        self,
        parent=None,
        default_name: str = "",
        default_expr: str = "",
        default_use_normalised: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Band Maths Expression")
        self.resize(800, 600)
        self.setMinimumSize(700, 500)
        layout = QVBoxLayout(self)

        # Instructions
        info = QTextEdit(self)
        info.setReadOnly(True)
        info.setMinimumHeight(150)
        info.setHtml(
            "<b>Band Maths Syntax</b><br>"
            "<ul>"
            "<li>Expressions use <b>Python arithmetic syntax</b>: <code>+</code>, <code>-</code>, <code>*</code>, <code>/</code>, <code>**</code>.</li>"
            "<li>To reference a spectral band (wavelength in nm), use the <b>R prefix</b>: <code>R780</code> or <code>R(780)</code>.</li>"
            "<li>Bare numeric literals (e.g., <code>1</code>, <code>0.5</code>, <code>1200</code>) are treated as <b>scalar constants</b>.</li>"
            "<li>The following safe function is available: <code>interp(R_w1, R_w2)</code> for linear interpolation.</li>"
            "</ul>"
            "Examples:"
            "<ul>"
            "<li><b>Al-OH Index (Clay):</b> <code>(R2200) / interp(R2100, R2300)</code></li>"
            "</ul>"
        
        
        )
        layout.addWidget(info)

        # Name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:", self))
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("e.g. Ratio 1")
        if default_name:
            self.name_edit.setText(default_name)
        name_row.addWidget(self.name_edit)
        layout.addLayout(name_row)

        # Expression
        expr_row = QHBoxLayout()
        expr_row.addWidget(QLabel("Expression:", self))
        self.expr_edit = QLineEdit(self)
        self.expr_edit.setPlaceholderText("e.g. R2300-R1400")
        if default_expr:
            self.expr_edit.setText(default_expr)
        expr_row.addWidget(self.expr_edit)
        layout.addLayout(expr_row)

        # Checkbox — evaluate on normalised spectra
        self.chk_norm = QCheckBox("Evaluate on normalised spectra?", self)
        self.chk_norm.setChecked(default_use_normalised)
        layout.addWidget(self.chk_norm)

        # Buttons (no custom slots)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.expr_edit.setFocus()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def get_expression(
        parent=None,
        default_name: str = "",
        default_expr: str = "",
        default_use_normalised: bool = True,
    ):
        """
        Open the dialog and return (ok, name, expr, cr)
        where cr indicates whether to use normalised spectra.
        """
        dlg = BandMathsDialog(
            parent=parent,
            default_name=default_name,
            default_expr=default_expr,
            default_use_normalised=default_use_normalised,
        )

        if dlg.exec_() != QDialog.Accepted:
            return False, "", "", False

        name = dlg.name_edit.text().strip()
        expr = dlg.expr_edit.text().strip()
        cr = dlg.chk_norm.isChecked()

        return True, name, expr, cr