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
from midas_gui.settings import Settings


class PythonHighlighter(QSyntaxHighlighter):
    """Minimal Python syntax highlighter for the code preview."""

    KEYWORDS = {
        "import", "from", "as", "def", "class", "return", "if", "else",
        "elif", "for", "while", "try", "except", "with", "None", "True",
        "False", "and", "or", "not", "in", "is", "lambda", "yield",
    }

    def highlightBlock(self, text: str):
        import re

        # 1. Find all string spans
        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor(THEME.success))
        string_spans: list[tuple[int, int]] = []
        for m in re.finditer(r"""("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""", text):
            span = (m.start(), m.end())
            string_spans.append(span)
            self.setFormat(span[0], span[1] - span[0], string_fmt)

        def _in_string(pos: int) -> bool:
            return any(s <= pos < e for s, e in string_spans)

        # 2. Find comment (first # not inside a string)
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor(THEME.text_secondary))
        comment_start = -1
        for i, c in enumerate(text):
            if c == "#" and not _in_string(i):
                comment_start = i
                self.setFormat(i, len(text) - i, comment_fmt)
                break

        # 3. Keywords — only in code regions (not in strings or comments)
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor(THEME.node_parameters))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        for m in re.finditer(r"\b(\w+)\b", text):
            if m.group(1) not in self.KEYWORDS:
                continue
            pos = m.start()
            if _in_string(pos):
                continue
            if comment_start >= 0 and pos >= comment_start:
                continue
            self.setFormat(pos, m.end() - pos, kw_fmt)


class CodePreview(QWidget):
    """Dockable panel showing the generated Python script."""

    def __init__(self, graph: GraphModel, settings: Settings, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._settings = settings

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
        self._apply_code_font()
        layout.addWidget(self.text_edit)

        self._highlighter = PythonHighlighter(self.text_edit.document())

        settings.font_size_changed.connect(self._apply_code_font)

    def _apply_code_font(self):
        size = self._settings.code_preview_font_size
        font = QFont("Cascadia Code", size, weight=QFont.Weight.Normal)
        if not QFont("Cascadia Code").exactMatch():
            font = QFont("Consolas", size)
        self.text_edit.setFont(font)

    def refresh(self):
        imported = getattr(self.parent(), '_imported_modules', None) if self.parent() else None
        code = generate_script(
            self.graph,
            runnable=self.runnable_check.isChecked(),
            comments=self.comments_check.isChecked(),
            imported_modules=imported,
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
