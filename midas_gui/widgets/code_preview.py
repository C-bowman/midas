from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
    QPushButton, QCheckBox, QFileDialog, QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat

from midas_gui.session import GraphModel
from midas_gui.script_generator import generate_script
from midas_gui.theme import THEME


class PythonHighlighter(QSyntaxHighlighter):
    """Minimal Python syntax highlighter for the code preview."""

    KEYWORDS = {
        "import", "from", "as", "def", "class", "return", "if", "else",
        "elif", "for", "while", "try", "except", "with", "None", "True",
        "False", "and", "or", "not", "in", "is", "lambda", "yield",
    }

    def highlightBlock(self, text: str):
        # Comments
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor(THEME.text_secondary))
        if "#" in text:
            idx = text.index("#")
            self.setFormat(idx, len(text) - idx, comment_fmt)

        # Strings
        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor(THEME.success))
        in_str = False
        quote_char = None
        start = 0
        for i, c in enumerate(text):
            if not in_str and c in ('"', "'"):
                in_str = True
                quote_char = c
                start = i
            elif in_str and c == quote_char:
                self.setFormat(start, i - start + 1, string_fmt)
                in_str = False

        # Keywords
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor(THEME.node_parameters))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        for word in text.split():
            clean = word.strip("(),:=[]")
            if clean in self.KEYWORDS:
                idx = text.find(clean)
                if idx >= 0:
                    self.setFormat(idx, len(clean), kw_fmt)


class CodePreview(QWidget):
    """Dockable panel showing the generated Python script."""

    def __init__(self, graph: GraphModel, parent=None):
        super().__init__(parent)
        self.graph = graph

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Generated Script")
        header.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {THEME.text_primary}; padding: 4px;"
        )
        layout.addWidget(header)

        # Options row
        opts = QHBoxLayout()
        self.runnable_check = QCheckBox("Runnable template")
        self.runnable_check.setToolTip("Add optimization and sampling stubs")
        self.runnable_check.toggled.connect(self.refresh)
        opts.addWidget(self.runnable_check)

        self.comments_check = QCheckBox("Include comments")
        self.comments_check.toggled.connect(self.refresh)
        opts.addWidget(self.comments_check)

        opts.addStretch()

        export_btn = QPushButton("Export .py")
        export_btn.clicked.connect(self._export)
        opts.addWidget(export_btn)
        layout.addLayout(opts)

        # Code display
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Cascadia Code", 10, weight=QFont.Weight.Normal))
        if not QFont("Cascadia Code").exactMatch():
            self.text_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(self.text_edit)

        self._highlighter = PythonHighlighter(self.text_edit.document())

    def refresh(self):
        code = generate_script(
            self.graph,
            runnable=self.runnable_check.isChecked(),
            comments=self.comments_check.isChecked(),
        )
        self.text_edit.setPlainText(code)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Python Script", "analysis.py",
            "Python files (*.py);;All files (*)",
        )
        if path:
            code = self.text_edit.toPlainText()
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
