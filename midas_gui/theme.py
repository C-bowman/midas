from dataclasses import dataclass
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


@dataclass(frozen=True)
class ThemeColors:
    # Meta
    name: str = "Deep Ocean"

    # Backgrounds
    bg_base: str = "#0F111A"
    bg_surface: str = "#1A1C2A"
    bg_elevated: str = "#232637"
    border: str = "#2B2F40"

    # Text
    text_primary: str = "#EEFFFF"
    text_secondary: str = "#717CB4"

    # Accents
    accent_primary: str = "#84FFFF"
    accent_secondary: str = "#82AAFF"

    # Status
    success: str = "#C3E88D"
    warning: str = "#FFCB6B"
    error: str = "#FF5370"

    # Node categories
    node_parameters: str = "#C792EA"
    node_data_inputs: str = "#FF5370"
    node_field_models: str = "#82AAFF"
    node_diagnostic_models: str = "#89DDFF"
    node_likelihoods: str = "#C3E88D"
    node_priors: str = "#FFCB6B"
    node_uncertainties: str = "#F78C6C"

    # Canvas
    canvas_grid: str = "#2B2F40"
    wire_color: str = "#82AAFF"
    wire_opacity: float = 0.8
    selected_border: str = "#84FFFF"


DEEP_OCEAN = ThemeColors()

DARK_2026 = ThemeColors(
    name="VSCode Dark",
    bg_base="#1E1E1E",
    bg_surface="#252526",
    bg_elevated="#2D2D2D",
    border="#3E3E3E",
    text_primary="#D4D4D4",
    text_secondary="#808080",
    accent_primary="#569CD6",
    accent_secondary="#4FC1FF",
    success="#6A9955",
    warning="#CCA700",
    error="#F44747",
    node_parameters="#C586C0",
    node_data_inputs="#F44747",
    node_field_models="#4FC1FF",
    node_diagnostic_models="#9CDCFE",
    node_likelihoods="#6A9955",
    node_priors="#DCDCAA",
    node_uncertainties="#CE9178",
    canvas_grid="#3E3E3E",
    wire_color="#4FC1FF",
    wire_opacity=0.8,
    selected_border="#569CD6",
)

LIGHT_2026 = ThemeColors(
    name="VSCode Light",
    bg_base="#FFFFFF",
    bg_surface="#F3F3F3",
    bg_elevated="#ECECEC",
    border="#CECECE",
    text_primary="#1E1E1E",
    text_secondary="#6A6A6A",
    accent_primary="#005FB8",
    accent_secondary="#0078D4",
    success="#388A34",
    warning="#BF8803",
    error="#CD3131",
    node_parameters="#AF00DB",
    node_data_inputs="#CD3131",
    node_field_models="#0078D4",
    node_diagnostic_models="#267F99",
    node_likelihoods="#388A34",
    node_priors="#795E26",
    node_uncertainties="#A31515",
    canvas_grid="#E0E0E0",
    wire_color="#0078D4",
    wire_opacity=0.85,
    selected_border="#005FB8",
)

THEMES: dict[str, ThemeColors] = {
    "Deep Ocean": DEEP_OCEAN,
    "VSCode Dark": DARK_2026,
    "VSCode Light": LIGHT_2026,
}

THEME: ThemeColors = DEEP_OCEAN

CATEGORY_COLORS: dict[str, str] = {}


def _rebuild_category_colors():
    """Rebuild CATEGORY_COLORS from the current THEME."""
    CATEGORY_COLORS.clear()
    CATEGORY_COLORS.update({
        "Data & Inputs": THEME.node_data_inputs,
        "Parameters & Fields": THEME.node_parameters,
        "Field Models": THEME.node_field_models,
        "Diagnostic Models": THEME.node_diagnostic_models,
        "Likelihoods": THEME.node_likelihoods,
        "Priors": THEME.node_priors,
        "Uncertainty Models": THEME.node_uncertainties,
    })


_rebuild_category_colors()


def set_theme(name: str):
    """Set the active theme by name. Must be called before apply_theme."""
    global THEME
    THEME = THEMES.get(name, DEEP_OCEAN)
    _rebuild_category_colors()


def apply_theme(app: QApplication):
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(THEME.bg_surface))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(THEME.text_primary))
    palette.setColor(QPalette.ColorRole.Base, QColor(THEME.bg_base))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(THEME.bg_elevated))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(THEME.bg_elevated))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(THEME.text_primary))
    palette.setColor(QPalette.ColorRole.Text, QColor(THEME.text_primary))
    palette.setColor(QPalette.ColorRole.Button, QColor(THEME.bg_elevated))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(THEME.text_primary))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(THEME.accent_primary))
    palette.setColor(QPalette.ColorRole.Link, QColor(THEME.accent_secondary))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(THEME.accent_secondary))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(THEME.bg_base))

    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
        QColor(THEME.text_secondary),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
        QColor(THEME.text_secondary),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
        QColor(THEME.text_secondary),
    )

    app.setPalette(palette)

    qss = f"""
    QMainWindow {{
        background-color: {THEME.bg_surface};
    }}
    QMenuBar {{
        background-color: {THEME.bg_surface};
        color: {THEME.text_primary};
        border-bottom: 1px solid {THEME.border};
    }}
    QMenuBar::item:selected {{
        background-color: {THEME.bg_elevated};
    }}
    QMenu {{
        background-color: {THEME.bg_elevated};
        color: {THEME.text_primary};
        border: 1px solid {THEME.border};
    }}
    QMenu::item:selected {{
        background-color: {THEME.accent_secondary};
        color: {THEME.bg_base};
    }}
    QDockWidget {{
        color: {THEME.text_primary};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {THEME.bg_surface};
        padding: 6px;
        border-bottom: 1px solid {THEME.border};
    }}
    QTreeWidget, QListWidget {{
        background-color: {THEME.bg_base};
        color: {THEME.text_primary};
        border: 1px solid {THEME.border};
        outline: none;
    }}
    QTreeWidget::item:hover, QListWidget::item:hover {{
        background-color: {THEME.bg_elevated};
    }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background-color: {THEME.accent_secondary};
        color: {THEME.bg_base};
    }}
    QHeaderView::section {{
        background-color: {THEME.bg_elevated};
        color: {THEME.text_primary};
        border: 1px solid {THEME.border};
        padding: 4px;
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {THEME.bg_base};
        color: {THEME.text_primary};
        border: 1px solid {THEME.border};
        border-radius: 3px;
        padding: 4px 8px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {THEME.accent_secondary};
    }}
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {THEME.bg_elevated};
        color: {THEME.text_primary};
        selection-background-color: {THEME.accent_secondary};
        selection-color: {THEME.bg_base};
    }}
    QPushButton {{
        background-color: {THEME.bg_elevated};
        color: {THEME.text_primary};
        border: 1px solid {THEME.border};
        border-radius: 4px;
        padding: 6px 16px;
    }}
    QPushButton:hover {{
        background-color: {THEME.border};
    }}
    QPushButton:pressed {{
        background-color: {THEME.accent_secondary};
        color: {THEME.bg_base};
    }}
    QScrollBar:vertical {{
        background-color: {THEME.bg_base};
        width: 10px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: {THEME.border};
        border-radius: 5px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {THEME.text_secondary};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background-color: {THEME.bg_base};
        height: 10px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {THEME.border};
        border-radius: 5px;
        min-width: 20px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: {THEME.text_secondary};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QStatusBar {{
        background-color: {THEME.bg_surface};
        color: {THEME.text_secondary};
        border-top: 1px solid {THEME.border};
    }}
    QLabel {{
        color: {THEME.text_primary};
    }}
    QGroupBox {{
        color: {THEME.text_primary};
        border: 1px solid {THEME.border};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0 4px;
    }}
    QSplitter::handle {{
        background-color: {THEME.border};
    }}
    QTextEdit, QPlainTextEdit {{
        background-color: {THEME.bg_base};
        color: {THEME.text_primary};
        border: 1px solid {THEME.border};
        font-family: 'Cascadia Code', 'Consolas', monospace;
    }}
    """
    app.setStyleSheet(qss)

    font = QFont("Segoe UI", 10)
    app.setFont(font)
