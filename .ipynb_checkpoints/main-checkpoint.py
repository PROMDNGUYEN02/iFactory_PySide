# app.py
from __future__ import annotations

import os
import sys
import logging
import getpass
import warnings
import configparser
import re
import traceback
from string import Template
from datetime import datetime
from typing import Any, Dict, List, Optional
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager, suppress

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QStatusBar, QWidget, QVBoxLayout,
    QLineEdit, QCheckBox, QPushButton, QHBoxLayout, QMenu,
    QWidgetAction, QListWidget, QListWidgetItem, QHeaderView, QTableView,
    QAbstractItemView
)
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QGuiApplication, QIcon
from PySide6.QtCore import (
    QTimer, QObject, QRunnable, QThreadPool, QPoint,
    Slot, QRegularExpression, QSortFilterProxyModel, Qt,
    QModelIndex, Signal, QAbstractTableModel
)

warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy")

# =========================================================
# ---- Logging Setup (singleton, no duplicate handlers) ----
# =========================================================
class AppLogger:
    _instance: Optional["AppLogger"] = None

    def __new__(cls, *_args, **_kwargs):
        """Ensure only one instance (Singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: str = "app",
        log_dir: str = "logs",
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.name = name
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        self._configure_encoding()
        self._ensure_log_dir()
        self._setup_handlers()

    def _configure_encoding(self):
        """Ensure UTF-8 console output encoding."""
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")  # noqa  # type: ignore[attr-defined]
            else:
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")  # type: ignore[assignment]
        except Exception:
            pass

    def _ensure_log_dir(self):
        """Create the log directory if it doesn't exist."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def _setup_handlers(self):
        """Configure file and console logging handlers."""
        if self.logger.handlers:
            return  # already configured

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
        )

        file_handler = RotatingFileHandler(
            os.path.join(self.log_dir, f"{self.name}.log"),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Return the configured logger."""
        return self.logger


if __name__ == "__main__":
    log = AppLogger().get_logger()
    log.info("Application started.")
    log.info(f"User: {getpass.getuser()} is running the application.")

# =========================================================
# ---- Config File Load ----
# =========================================================
class ConfigLoader:
    def __init__(self, config_path="config.ini"):
        self.config_path = Path(config_path)
        self.parser = configparser.ConfigParser()
        self.parser.optionxform = str  # type: ignore[assignment]
        self.config: Optional[SimpleNamespace] = None

        self.required_config = {
            "iFactory": ["app_name", "version", "developer", "icon_path"],
            "SplashScreen": ["background_path", "logo_path"],
            "Database": ["host", "user", "database"],
            "UI": ["light", "dark"]
        }

    def load(self) -> SimpleNamespace:
        self._check_file_exists()
        self._read_config()
        self._validate_config()
        self._to_namespace()
        return self.config  # type: ignore[return-value]

    def _check_file_exists(self):
        if not self.config_path.is_file():
            log.error(f"Configuration file '{self.config_path}' not found.")
            raise FileNotFoundError(f"Missing configuration file: {self.config_path}")

    def _read_config(self):
        try:
            with self.config_path.open(encoding="utf-8") as f:
                self.parser.read_file(f)
            log.info("Configuration loaded successfully.")
        except configparser.Error as e:
            log.error(f"Unable to read config file: {e}")
            raise RuntimeError(f"Unable to read config file: {e}")

    def _validate_config(self):
        log.info("Validating configuration file...")
        if not self.parser.sections():
            raise RuntimeError("Configuration file is empty or missing sections.")

        missing_items: List[str] = []
        for section, keys in self.required_config.items():
            if section not in self.parser:
                missing_items.append(f"[{section}] (entire section missing)")
                continue
            for key in keys:
                if key not in self.parser[section]:
                    missing_items.append(f"[{section}] {key}")

        if missing_items:
            raise RuntimeError("Missing configuration entries:\n - " + "\n - ".join(missing_items))

    def _to_namespace(self):
        self.config = SimpleNamespace(**{
            section: SimpleNamespace(**dict(self.parser.items(section)))
            for section in self.parser.sections()
        })


# =========================================================
# ---- DB Connection Helpers ----
# =========================================================
try:
    import pyodbc
    log.info("pyodbc module loaded successfully.")
except ImportError as e:
    log.error("pyodbc module is not installed. Please install it using 'pip install pyodbc'.")
    raise RuntimeError("pyodbc module is not installed.") from e


class SqlServerConnector:
    """Handles SQL Server connection using pyodbc and config settings."""

    PREFERRED_DRIVERS = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server"
    ]

    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.driver = self._get_odbc_driver()

    def _get_odbc_driver(self) -> str:
        available_drivers = pyodbc.drivers()
        if not available_drivers:
            log.error("No ODBC drivers found. Please install an ODBC driver for SQL Server.")
            raise RuntimeError("No ODBC drivers found. Please install an ODBC driver for SQL Server.")
        log.info(f"Available ODBC drivers: {available_drivers}")

        for driver in self.PREFERRED_DRIVERS:
            if driver in available_drivers:
                log.info(f"Using ODBC driver: {driver}")
                return driver

        log.error("No suitable ODBC driver found for SQL Server.")
        raise RuntimeError("No suitable ODBC driver found for SQL Server.")

    @contextmanager
    def get_connection(self):
        """Create and return a pyodbc connection."""
        db: SimpleNamespace = self.config.Database
        password = getattr(db, "password", None) or os.getenv("DB_PASSWORD", "")
        if not password:
            log.warning("Database password not found in config or env (DB_PASSWORD). Attempting connection without PWD.")

        conn_str = (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={db.host};"
            f"DATABASE={db.database};"
            f"UID={db.user};"
            f"{'PWD=' + password + ';' if password else ''}"
            f"TrustServerCertificate=yes;"
        )

        conn = None
        try:
            conn = pyodbc.connect(conn_str, timeout=5)
            log.info("Database connection established successfully.")
            yield conn
        except pyodbc.Error as ex:
            log.error(f"Database connection failed: {ex.args}")
            raise RuntimeError(f"Database connection failed: {ex}") from ex
        finally:
            with suppress(Exception):
                if conn is not None:
                    conn.close()


# =========================================================
# ---- Data Model Layer ----
# =========================================================
log.info("Checking pandas installation...")
try:
    import pandas as pd
except ImportError as e:
    log.error("pandas module is not installed. Please install it using 'pip install pandas'.")
    raise RuntimeError("pandas module is not installed.") from e
if not hasattr(pd, "read_sql"):
    log.error("pandas does not support SQLAlchemy/DB-API read_sql.")
    raise RuntimeError("pandas does not support SQLAlchemy/DB-API read_sql.")


class PandasModel(QAbstractTableModel):
    """Qt model for displaying a pandas DataFrame in QTableView with in-place editing."""
    def __init__(self, df: pd.DataFrame, parent: Optional[QObject] = None):
        super().__init__(parent)
        # keep only one copy; editing mutates _df
        self._original_df = df.copy(deep=False)
        self._df = df.copy(deep=False)
        self.filtered_columns = set()
        self.filter_icon = QIcon.fromTheme("view-filter")

    # ---------- Dimensions ----------
    def rowCount(self, _=QModelIndex()): return len(self._df)
    def columnCount(self, _=QModelIndex()): return len(self._df.columns)

    # ---------- Data Access ----------
    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(val) else str(val)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(self, section, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self._df.columns[section])
            elif role == Qt.ItemDataRole.DecorationRole and section in self.filtered_columns:
                return self.filter_icon
        return super().headerData(section, orientation, role)

    # ---------- Data Modification ----------
    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        try:
            self._df.iat[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, [role])
            return True
        except Exception as e:
            log.error(f"Error setting data: {e}")
            return False

    def setHeaderData(self, section, orientation, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole:
            return False
        try:
            if orientation == Qt.Orientation.Horizontal:
                self._df.columns.values[section] = value
            else:
                self._df.index.values[section] = value
            self.headerDataChanged.emit(orientation, section, section)
            return True
        except Exception as e:
            log.error(f"Error setting header: {e}")
            return False

    def update_data(self, df: pd.DataFrame):
        self.beginResetModel()
        self._original_df = df.copy(deep=False)
        self._df = df.copy(deep=False)
        self.endResetModel()

    # ---------- Sort & Filter Reset ----------
    def clear_sort(self):
        self.beginResetModel()
        self._df = self._original_df
        self.endResetModel()

    def clear_filters(self):
        self.filtered_columns.clear()
        self.beginResetModel()
        self._df = self._original_df
        self.endResetModel()

    # ---------- Flags ----------
    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag(0)
        return (Qt.ItemFlag.ItemIsSelectable |
                Qt.ItemFlag.ItemIsEnabled |
                Qt.ItemFlag.ItemIsEditable)


# =========================================================
# ---- Filtering Proxy Model (precompiled regex) ----
# =========================================================
class MultiColumnFilterProxy(QSortFilterProxyModel):
    """Supports global text search and per-column regex filtering."""
    filtersChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.global_terms: List[str] = []
        self.global_regexes: List[QRegularExpression] = []
        self.column_filters: Dict[int, QRegularExpression] = {}

    def setGlobalFilterTerms(self, terms: List[str]):
        self.global_terms = terms or []
        self.global_regexes = [
            QRegularExpression(re.escape(t), QRegularExpression.PatternOption.CaseInsensitiveOption)
            for t in self.global_terms
        ]
        self._update_filters()

    def setColumnFilter(self, column: int, regex: QRegularExpression):
        if regex and regex.pattern():
            self.column_filters[column] = regex
        else:
            self.column_filters.pop(column, None)
        self._update_filters()

    def clearAllColumnFilters(self):
        self.column_filters.clear()
        self._update_filters()

    def _update_filters(self):
        self.invalidateFilter()
        self.filtersChanged.emit()

    def filterAcceptsRow(self, source_row: int, parent: QModelIndex) -> bool:
        model = self.sourceModel()
        col_count = model.columnCount()
        # Global filter: each term must match somewhere in the row
        for regex in self.global_regexes:
            # any column matches?
            if not any(regex.match(str(model.data(model.index(source_row, c, parent)))).hasMatch()
                       for c in range(col_count)):
                return False
        # Per-column filters
        for c, regex in self.column_filters.items():
            if not regex.match(str(model.data(model.index(source_row, c, parent)))).hasMatch():
                return False
        return True


# =========================================================
# ---- Column Filter Menu & Header ----
# =========================================================
class ColumnFilterWidget(QWidget):
    filterApplied = Signal(int, QRegularExpression, list)

    def __init__(self, col, values, selected_values=None, search_text="", menu=None, parent=None):
        super().__init__(parent)
        self.col = col
        self.values = sorted(values)
        self.selected_values = set(selected_values or self.values)
        self.search_text = search_text
        self.menu = menu
        self._build_ui()
        self._connect_signals()
        self._update_select_all()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Search box
        self.search_box = QLineEdit(placeholderText="Search...")
        self.search_box.setText(self.search_text)
        layout.addWidget(self.search_box)

        # Select all
        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.setTristate(True)
        layout.addWidget(self.select_all_cb)

        # List widget for items
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_widget.setMaximumHeight(200)
        for val in self.values:
            item = QListWidgetItem(val)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if val in self.selected_values else Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.select_visible_btn = QPushButton("Select Visible")
        self.clear_all_btn = QPushButton("Clear All")
        self.invert_btn = QPushButton("Invert")
        btn_layout.addWidget(self.select_visible_btn)
        btn_layout.addWidget(self.clear_all_btn)
        btn_layout.addWidget(self.invert_btn)
        layout.addLayout(btn_layout)

        # OK / Cancel
        btn_layout2 = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout2.addWidget(self.ok_btn)
        btn_layout2.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout2)

    def _connect_signals(self):
        self.list_widget.itemChanged.connect(self._update_select_all)
        self.select_all_cb.clicked.connect(
            lambda: self._toggle_all(Qt.CheckState.Checked if self.select_all_cb.isChecked()
                                     else Qt.CheckState.Unchecked)
        )
        self.search_box.textChanged.connect(self._filter_visible)
        self.select_visible_btn.clicked.connect(self._select_visible)
        self.clear_all_btn.clicked.connect(lambda: self._toggle_all(Qt.CheckState.Unchecked))
        self.invert_btn.clicked.connect(self._invert_selection)
        self.ok_btn.clicked.connect(self._apply)
        self.cancel_btn.clicked.connect(lambda: self.menu.close() if self.menu else None)

    def _update_select_all(self):
        checked = sum(self.list_widget.item(i).checkState() == Qt.CheckState.Checked
                      for i in range(self.list_widget.count()))
        if checked == self.list_widget.count():
            self.select_all_cb.setCheckState(Qt.CheckState.Checked)
        elif checked == 0:
            self.select_all_cb.setCheckState(Qt.CheckState.Unchecked)
        else:
            self.select_all_cb.setCheckState(Qt.CheckState.PartiallyChecked)

    def _toggle_all(self, state: Qt.CheckState):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def _select_visible(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)

    def _invert_selection(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                               else Qt.CheckState.Checked)

    def _filter_visible(self, text: str):
        lower = text.lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(lower not in item.text().lower())

    def _apply(self):
        selected = [self.list_widget.item(i).text() for i in range(self.list_widget.count())
                    if self.list_widget.item(i).checkState() == Qt.CheckState.Checked]
        pattern = f"^(?:{'|'.join(QRegularExpression.escape(v) for v in selected)})$" if selected else r"(?!)"
        regex = QRegularExpression(pattern)
        self.filterApplied.emit(self.col, regex, selected)
        if self.menu:
            self.menu.close()


class InteractiveHeader(QHeaderView):
    filterChanged = Signal()

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setSectionsClickable(True)
        self.last_selected_values: Dict[int, List[str]] = {}
        self.last_search_texts: Dict[int, str] = {}

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            col = self.logicalIndexAt(event.position().toPoint())
            if col >= 0:
                self.show_menu(col)
        super().mousePressEvent(event)

    def show_menu(self, col: int):
        table = self.parent()
        if not isinstance(table, QTableView):
            return
        model = table.model()
        src_model = model.sourceModel() if isinstance(model, QSortFilterProxyModel) else model

        values = sorted({str(src_model.data(src_model.index(r, col))) for r in range(src_model.rowCount())})
        selected = self.last_selected_values.get(col, values)
        search_text = self.last_search_texts.get(col, "")

        menu = QMenu(self)
        menu.addAction("⬆ Sort Ascending", lambda: self._sort(table, col, Qt.SortOrder.AscendingOrder))
        menu.addAction("⬇ Sort Descending", lambda: self._sort(table, col, Qt.SortOrder.DescendingOrder))
        menu.addSeparator()
        menu.addAction("❌ Clear Sort", lambda: self._clear_sort(table))
        if isinstance(model, QSortFilterProxyModel):
            menu.addAction("🧹 Clear All Filters", lambda: self._clear_filters(model, menu))

        filter_widget = ColumnFilterWidget(col, values, selected, search_text, menu)
        if isinstance(model, QSortFilterProxyModel):
            filter_widget.filterApplied.connect(
            lambda c, regex, sel: self._apply_filter(c, regex, sel, model, filter_widget.search_box.text())
            )
        fw_action = QWidgetAction(self)
        fw_action.setDefaultWidget(filter_widget)
        menu.addAction(fw_action)
        menu.exec(self.mapToGlobal(QPoint(self.sectionPosition(col), self.height())))

    def _sort(self, table: QTableView, col: int, order: Qt.SortOrder):
        table.sortByColumn(col, order)

    def _apply_filter(self, col: int, regex: QRegularExpression, selected_values: List[str],
                        model: QSortFilterProxyModel, search_text: str):
        if isinstance(model, MultiColumnFilterProxy):
            model.setColumnFilter(col, regex)
        self.last_selected_values[col] = selected_values
        self.last_search_texts[col] = search_text
        self.filterChanged.emit()

    def _clear_sort(self, table: QTableView):
        proxy = table.model()
        if isinstance(proxy, QSortFilterProxyModel):
            proxy.sort(-1)
            proxy.invalidate()
        table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

    def _clear_filters(self, model: QSortFilterProxyModel, menu: QMenu):
        if isinstance(model, MultiColumnFilterProxy):
            model.clearAllColumnFilters()
        self.last_selected_values.clear()
        self.last_search_texts.clear()
        self.filterChanged.emit()
        menu.close()


# =========================================================
# ---- Data Loading Worker (cancel outdated jobs) ----
# =========================================================
class DataLoadWorker(QObject, QRunnable):
    finished = Signal(str, object)
    error = Signal(str, str)

    def __init__(self, query_id: str, query_func, query: str):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.query_id = query_id
        self.query_func = query_func
        self.query = query
        self._cancelled = False
        self.setAutoDelete(False)

    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        if self._cancelled:
            return
        try:
            log.debug(f"Worker started for query_id: {self.query_id}")
            df = self.query_func(self.query)
            if self._cancelled:
                return
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Query did not return a DataFrame")
            log.debug(f"Worker finished successfully for query_id: {self.query_id}")
            self.finished.emit(self.query_id, df)
        except Exception:
            err_trace = traceback.format_exc()
            log.error(f"Worker encountered error for query_id {self.query_id}:\n{err_trace}")
            self.error.emit(self.query_id, err_trace)


# =========================================================
# ---- Table Viewer ----
# =========================================================
class TableViewer(QWidget):
    """Filterable, sortable table widget with optional auto-refresh."""
    loadRequested = Signal(str)

    def __init__(self, query: str, main_window: QMainWindow, query_id: str, query_func, refresh_ms: int = 10_000):
        super().__init__()
        self.query = query
        self.main_window = main_window
        self.query_id = query_id
        self.query_func = query_func
        self.is_loading = False

        # Models
        self.df_source = pd.DataFrame()
        self.source_model = PandasModel(self.df_source)
        self.proxy_model = MultiColumnFilterProxy()
        self.proxy_model.setSourceModel(self.source_model)

        # Auto-refresh timer (default 10s; adjust via config/UI as needed)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.request_load_data)
        self.refresh_timer.start(max(1000, refresh_ms))

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Toolbar
        bar = QHBoxLayout()
        self.filter_input = QLineEdit(placeholderText="🔍 Search all columns...")
        self.refresh_btn = QPushButton("⟳ Refresh")
        self.export_btn = QPushButton("⭳ Export")

        bar.addWidget(self.filter_input)
        bar.addStretch()
        bar.addWidget(self.refresh_btn)
        bar.addWidget(self.export_btn)
        layout.addLayout(bar)

        # Table
        self.table = QTableView()
        self.table.setModel(self.proxy_model)
        header = InteractiveHeader(Qt.Orientation.Horizontal, self.table)
        self.table.setHorizontalHeader(header)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

        # Footer
        self.status_label = QLabel("Waiting for data...")
        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(self.status_label)
        footer.addStretch()
        layout.addLayout(footer)

        # Signals
        self.refresh_btn.clicked.connect(self.request_load_data)
        self.export_btn.clicked.connect(self.export_data)
        self.filter_input.textChanged.connect(self.apply_global_filter)

    def request_load_data(self):
        if self.is_loading:
            return
        self.is_loading = True
        self.loadRequested.emit(self.query_id)

    def apply_global_filter(self):
        terms = [t.strip() for t in self.filter_input.text().split() if t.strip()]
        self.proxy_model.setGlobalFilterTerms(terms)

    def on_load_success(self, df: pd.DataFrame):
        self.is_loading = False
        self.df_source = df.copy(deep=False)
        self.source_model.update_data(self.df_source)
        self.status_label.setText(f"Loaded {len(df)} rows.")
        self.filter_input.clear()
        self.proxy_model.setGlobalFilterTerms([])

    def on_load_error(self, error_msg: str):
        self.is_loading = False
        self.status_label.setText(f"Error loading data: {error_msg}")
        QMessageBox.critical(self, "Data Load Error", error_msg)

    def export_data(self):
        # Export the *filtered & sorted* view by mapping proxy rows to source rows
        if self.proxy_model.rowCount() == 0:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
        query_id = getattr(self, "query_id", "default_query")
        path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", f"{query_id}_data.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        try:
            src_df = self.source_model._df
            # Determine visible rows in source order
            src_rows: List[int] = []
            for r in range(self.proxy_model.rowCount()):
                src_index = self.proxy_model.mapToSource(self.proxy_model.index(r, 0))
                src_rows.append(src_index.row())
            out_df = src_df.iloc[src_rows].reset_index(drop=True)
            out_df.to_excel(path, index=False)
            QMessageBox.information(self, "Export Successful", f"Data exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))


# =========================================================
# ---- Splash Screen ----
# =========================================================
class SplashScreen(QWidget):
    def __init__(self, splash_img_path: str, logo_img_path: str, title: str):
        super().__init__()
        self.setFixedSize(600, 400)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.splash_img = QPixmap(splash_img_path)

        logo = QLabel()
        logo.setPixmap(QPixmap(logo_img_path).scaledToWidth(160, Qt.TransformationMode.SmoothTransformation))

        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("color:white; font-size:22px; font-weight:bold;")

        self.status = QLabel("Initializing...")
        self.status.setStyleSheet("color:white; font-size:12px;")

        self.percent = QLabel("0%")
        self.percent.setStyleSheet("color:white; font-size:12px; font-weight:bold;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar { background: #2c2c2c; border: none; border-radius: 4px; height: 4px; }
            QProgressBar::chunk { background: #4caf50; border-radius: 4px; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        status_row = QHBoxLayout()
        status_row.addWidget(self.status)
        status_row.addStretch()
        status_row.addWidget(self.percent)

        main_layout.addWidget(logo)
        main_layout.addStretch()
        main_layout.addWidget(self.title)
        main_layout.addLayout(status_row)
        main_layout.addWidget(self.progress)

    def showEvent(self, event):
        self.move(QGuiApplication.primaryScreen().availableGeometry().center() - self.rect().center())
        super().showEvent(event)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(self.rect(), 20, 20)
        painter.setClipPath(path)

        scaled = self.splash_img.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled
        )

    def update_progress(self, message: str, percent: int):
        self.status.setText(message)
        self.percent.setText(f"{percent}%")
        self.progress.setValue(percent)


# =========================================================
# ---- Main Application ----
# =========================================================
class App(QMainWindow):
    """Main application window and startup controller."""
    def __init__(self, config_data: SimpleNamespace, db_connector: Optional[SqlServerConnector] = None):
        super().__init__()

        self.config = config_data
        self.db_connector = db_connector or SqlServerConnector(config_data)

        # State tracking
        self.tabs: Dict[str, TableViewer] = {}
        self.initial_loads_pending: set[str] = set()
        self.total_initial_loads = 0
        self._initial_load_completed = False
        self.active_workers: Dict[str, DataLoadWorker] = {}

        # UI
        self.setWindowIcon(QIcon(self.config.iFactory.icon_path))
        self.setStatusBar(QStatusBar(self))

        # Splash screen
        ss = getattr(self.config, "SplashScreen", SimpleNamespace())
        splash_bg = getattr(ss, "background_path", "")
        splash_logo = getattr(ss, "logo_path", "")
        if not splash_bg or not Path(splash_bg).exists():
            log.warning("SplashScreen.background_path missing or not found; splash may render blank.")
        if not splash_logo or not Path(splash_logo).exists():
            log.warning("SplashScreen.logo_path missing or not found; splash may render blank.")
        splash_text = f"{self.config.iFactory.app_name} {self.config.iFactory.version}"
        self.splash = SplashScreen(splash_bg, splash_logo, splash_text)

        # Style & Clock timers
        self._style_timer = QTimer(self)
        self._style_timer.timeout.connect(self.apply_styles)
        self._style_timer.start(15 * 60 * 1000)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._refresh_datetime)
        self._clock_timer.start(1000)

        # Limit global thread pool
        QThreadPool.globalInstance().setMaxThreadCount(4)

    # ------------------ Application startup ------------------
    def run(self):
        self.splash.show()
        QTimer.singleShot(100, self.load_sequence)
        if app := QApplication.instance():
            return app.exec()
        raise RuntimeError("QApplication instance not found.")

    def load_sequence(self):
        try:
            # Step 1: Init UI
            self._smooth_progress_to(5, "Initializing UI...")
            self.init_ui_structure()

            # Step 2: Connect DB (sanity check)
            self._smooth_progress_to(10, "Connecting to database...")
            with self.db_connector.get_connection():
                log.info("Initial database connection successful.")

            # Step 3: Start loading data
            self._smooth_progress_to(15, "Loading initial data...")
            self.start_initial_data_load()

        except Exception as e:
            log.critical(f"Startup sequence failed: {e}")
            QMessageBox.critical(self, "Startup Failed", str(e))
            self.close()

    def init_ui_structure(self):
        self.setWindowTitle(
            f"{self.config.iFactory.app_name} {self.config.iFactory.version} | "
            f"User: {getpass.getuser()} | Developed by {self.config.iFactory.developer}"
        )
        self.setContentsMargins(4, 0, 4, 0)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        queries = getattr(self.config, "Query", SimpleNamespace())
        refresh_ms = int(getattr(getattr(self.config, "App", SimpleNamespace()), "refresh_ms", "10000") or "10000")
        for query_id, query_str in vars(queries).items():
            viewer = TableViewer(
                query=query_str,
                main_window=self,
                query_id=query_id,
                query_func=self.query_function,
                refresh_ms=refresh_ms
            )
            viewer.loadRequested.connect(self.handle_data_load_request)
            self.tab_widget.addTab(viewer, query_id.replace('_', ' ').title())

            self.tabs[query_id] = viewer
            self.initial_loads_pending.add(query_id)

        self.total_initial_loads = len(self.initial_loads_pending)
        self.apply_styles()
        self.setup_datetime_clock()

    def start_initial_data_load(self):
        if not self.tabs:
            self.check_initial_load_complete("N/A")
            return
        for query_id in list(self.initial_loads_pending):
            self.handle_data_load_request(query_id)

    def query_function(self, query: str) -> pd.DataFrame:
        if not self.db_connector:
            raise RuntimeError("No database connector configured.")
        with self.db_connector.get_connection() as conn:
            conn_any: Any = conn  # type: ignore
            df = pd.read_sql_query(query, conn_any)
            return df

    # ------------------ Data loading handlers ------------------
    @Slot(str)
    def handle_data_load_request(self, query_id: str):
        viewer = self.tabs.get(query_id)
        if not viewer:
            log.error(f"No tab found for query_id: {query_id}")
            return

        if viewer.is_loading:
            log.debug(f"Load already in progress for {query_id}, skipping duplicate request.")
            return
        viewer.is_loading = True

        # cancel & remove any existing worker for this query_id
        old = self.active_workers.get(query_id)
        if old:
            old.cancel()
            try:
                old.finished.disconnect()
                old.error.disconnect()
            except Exception:
                pass
            self.active_workers.pop(query_id, None)

        worker = DataLoadWorker(query_id, viewer.query_func, viewer.query)
        worker.finished.connect(self.on_data_load_finished)
        worker.error.connect(self.on_data_load_error)
        self.active_workers[query_id] = worker
        QThreadPool.globalInstance().start(worker)

    @Slot(str, pd.DataFrame)
    def on_data_load_finished(self, query_id: str, df: pd.DataFrame):
        viewer = self.tabs.get(query_id)
        if viewer:
            viewer.is_loading = False
            log.debug(f"Updating UI for {query_id} with new data")
            try:
                viewer.on_load_success(df)
            except Exception:
                log.error(f"Error updating viewer for {query_id}:\n{traceback.format_exc()}")

        # cleanup worker
        worker = self.active_workers.pop(query_id, None)
        if worker:
            with suppress(Exception):
                worker.finished.disconnect()
            with suppress(Exception):
                worker.error.disconnect()

        self.check_initial_load_complete(query_id)

    @Slot(str, str)
    def on_data_load_error(self, query_id: str, error_msg: str):
        log.error(f"Data load failed for {query_id}:\n{error_msg}")
        viewer = self.tabs.get(query_id)
        if viewer:
            viewer.is_loading = False
            try:
                viewer.on_load_error(error_msg)
            except Exception:
                log.error(f"Error handling load error in viewer for {query_id}:\n{traceback.format_exc()}")

        # cleanup worker
        worker = self.active_workers.pop(query_id, None)
        if worker:
            with suppress(Exception):
                worker.finished.disconnect()
            with suppress(Exception):
                worker.error.disconnect()

    # ------------------ Progress tracking ------------------
    def check_initial_load_complete(self, query_id: str):
        self.initial_loads_pending.discard(query_id)
        loads_done = self.total_initial_loads - len(self.initial_loads_pending)

        target_progress = 15 + int((loads_done / self.total_initial_loads) * 45) if self.total_initial_loads > 0 else 60
        label = f"Loaded {query_id} ({loads_done}/{self.total_initial_loads})..."
        self._smooth_progress_to(target_progress, label)

        if not self.initial_loads_pending and not self._initial_load_completed:
            self._smooth_progress_to(90, "All data loaded.")
            QTimer.singleShot(200, self.show_main_window)
            self._initial_load_completed = True

    def _smooth_progress_to(self, target: int, label: str):
        """Animate progress bar with a ~60 FPS cap to reduce CPU."""
        self._progress_target = max(0, min(100, target))
        self._progress_label = label

        if not hasattr(self, "_progress_timer"):
            self._progress_timer = QTimer(self)
            self._progress_timer.timeout.connect(self._progress_step)
        if self._progress_timer.isActive():
            self._progress_timer.stop()
        self._progress_timer.start(16)  # ~60 FPS

    def _progress_step(self):
        current = self.splash.progress.value()
        step = 1 if current < self._progress_target else (-1 if current > self._progress_target else 0)
        if step != 0:
            self.splash.update_progress(self._progress_label, current + step)
        else:
            self._progress_timer.stop()

    # ------------------ UI helpers ------------------
    def apply_styles(self):
        hour = datetime.now().hour
        theme = "light" if 6 <= hour < 18 else "dark"

        # Colors
        colors_parent = getattr(self.config, "Colors", None)
        theme_colors = getattr(colors_parent, theme, {}) if colors_parent else {}
        colors = dict(theme_colors) if isinstance(theme_colors, dict) else {}
        if not colors:
            logging.warning(f"No colors found for theme '{theme}'.")

        # UI style
        ui_parent = getattr(self.config, "UI", None)
        raw_style = getattr(ui_parent, theme, "") if ui_parent else ""
        if not raw_style:
            logging.warning(f"No UI stylesheet found for theme '{theme}'.")

        # Apply style
        try:
            style = Template(raw_style).safe_substitute(colors)
            self.setStyleSheet(style)
            logging.info(f"Applied '{theme}' theme successfully.")
        except Exception as e:
            logging.error(f"Error applying '{theme}' theme: {e}")

    @property
    def current_table_viewer(self) -> Optional[TableViewer]:
        widget = self.tab_widget.currentWidget()
        return widget if isinstance(widget, TableViewer) else None

    def show_main_window(self):
        self.splash.close()
        self.resize(1280, 768)
        self.show()
        self.statusBar().showMessage("Application ready.", 3000)
        log.info("Main window displayed. Application is ready.")

    # ------------------ Datetime clock ------------------
    def setup_datetime_clock(self):
        self._datetime_label = QLabel(alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 10, 5)
        layout.addWidget(self._datetime_label)
        self.tab_widget.setCornerWidget(container, Qt.Corner.TopRightCorner)
        self._refresh_datetime()

    def _refresh_datetime(self):
        self._datetime_label.setText(datetime.now().strftime("🕒 %Y-%m-%d %H:%M:%S"))


# =========================================================
# ---- Entrypoint ----
# =========================================================
if __name__ == '__main__':
    # Load configuration
    loader = ConfigLoader("config.ini")
    config = loader.load()

    # Start QApplication
    app = QApplication(sys.argv)
    if config is None:
        raise RuntimeError("Configuration failed to load and is None.")
    main_app = App(config)
    sys.exit(main_app.run())