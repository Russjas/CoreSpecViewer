# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:28:13 2025

@author: russj
"""


from pathlib import Path
from datetime import datetime
import csv, threading, traceback


from objects import RawObject
from tools import discover_lumo_directories,crop_auto


from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QFileDialog
)


# ----------------------------
# Thread-safe CSV log
# ----------------------------
class ProcessingLogger:
    """Thread-safe CSV logger that writes <dest>/processing_log.csv."""
    def __init__(self, dest_dir: Path, filename: str = "processing_log.csv"):
        self.path = Path(dest_dir) / filename
        self.lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with self.path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "timestamp", "box_path", "status", "duration_s",
                        "output_dir", "message"
                    ])
                    w.writeheader()

    def write(self, **row):
        row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        with self.lock, self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp", "box_path", "status", "duration_s",
                "output_dir", "message"
            ])
            w.writerow(row)


# ----------------------------
# Core batch processor (no UI)
# ----------------------------
def process_multibox(src, dest, *,
                     logger: ProcessingLogger | None = None,
                     force: bool = False,
                     progress_cb=None,             # callable(done, total, msg)
                     should_stop=None):            # callable() -> bool
    """
    Batch process Lumo-like directories under `src` into `dest`.
    Writes a CSV log to <dest>/processing_log.csv by default.

    Returns a tuple (attempted, ok, skipped, errors, not_lumo).
    """
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    logger = logger or ProcessingLogger(dest)

    dirs = discover_lumo_directories(src)
    total = len(dirs)
    done = 0
    stats = {"ok": 0, "skipped": 0, "error": 0, "not_lumo": 0}

    for box in dirs:
        if should_stop and should_stop():
            logger.write(box_path=str(box), status="aborted",
                         duration_s="", output_dir=str(dest),
                         message="User cancelled")
            if progress_cb:
                progress_cb(done, total, f"aborted at {box.name}")
            break

        t0 = datetime.now()
        msg = ""
        status = "ok"

        try:
            # Skip if already processed (simple rule: metadata.json exists)
            out_meta = dest / f"{box.name}_metadata.json"
            if out_meta.exists() and not force:
                status, msg = "skipped", "already processed"
            else:
                raw = RawObject.from_Lumo_directory(box)     # may raise ValueError
                cropped = crop_auto(raw)
                po = cropped.process()
                po.update_root_dir(dest)
                po.save_all()
                out_meta = str(po.datasets['metadata'].path)
                msg = "processed ok"

        except ValueError as e:
            status, msg = "not_lumo", f"{e}"
        except Exception:
            status = "error"
            msg = traceback.format_exc(limit=2).strip().replace("\n", " | ")

        duration = (datetime.now() - t0).total_seconds()
        logged_path = out_meta if 'out_meta' in locals() and Path(out_meta).exists() else dest
        logger.write(box_path=str(box), status=status, duration_s=duration,
                     output_dir=str(dest), message=msg)

        stats[status] = stats.get(status, 0) + 1
        done += 1
        if progress_cb:
            progress_cb(done, total, f"{box.name}: {msg}")

    attempted = done
    return attempted, stats["ok"], stats["skipped"], stats["error"], stats["not_lumo"]


# ----------------------------
# Lightweight QRunnable worker
# ----------------------------
class MultiboxSignals(QObject):
    progress = pyqtSignal(int, int, str)          # done, total, message
    finished = pyqtSignal(tuple)                  # (attempted, ok, skipped, error, not_lumo)
    cancelled = pyqtSignal()


class MultiboxWorker(QRunnable):
    def __init__(self, src, dest, force=False):
        super().__init__()
        self.src = src
        self.dest = dest
        self.force = force
        self.signals = MultiboxSignals()
        self._stop = False

    def request_cancel(self):
        self._stop = True

    def run(self):
        def progress_cb(done, total, msg):
            self.signals.progress.emit(done, total, msg)

        def should_stop():
            return self._stop

        result = process_multibox(
            self.src, self.dest,
            logger=None,
            force=self.force,
            progress_cb=progress_cb,
            should_stop=should_stop
        )
        if self._stop:
            self.signals.cancelled.emit()
        else:
            self.signals.finished.emit(result)


def run_multibox_dialog(parent):
    """
    Opens folder pickers, starts background processing, and shows a non-modal
    progress dialog. Keeps all UI and threading *out* of MainWindow.
    """
    src = QFileDialog.getExistingDirectory(parent, "Select hole/source folder")
    if not src:
        return
    dest = QFileDialog.getExistingDirectory(parent, "Select destination folder")
    if not dest:
        return

    dlg = QDialog(parent)
    dlg.setWindowTitle("Processing boxes…")
    dlg.setModal(False)

    v = QVBoxLayout(dlg)
    lbl = QLabel("Starting…")
    bar = QProgressBar()
    bar.setRange(0, 0)  # set to 0..N on first progress
    cancel_btn = QPushButton("Cancel")
    v.addWidget(lbl); v.addWidget(bar); v.addWidget(cancel_btn)
    dlg.resize(420, 130)
    dlg.show()

    worker = MultiboxWorker(src, dest, force=False)

    def on_progress(done, total, msg):
        if bar.maximum() == 0:
            bar.setRange(0, total)
        bar.setValue(done)
        lbl.setText(msg)

    def on_finished(result_tuple):
        attempted, ok, skipped, err, not_lumo = result_tuple
        lbl.setText(f"Done. ok={ok}, skipped={skipped}, errors={err}, not_lumo={not_lumo}")
        bar.setValue(bar.maximum())
        cancel_btn.setEnabled(False)

    def on_cancelled():
        lbl.setText("Cancelled by user.")
        cancel_btn.setEnabled(False)

    cancel_btn.clicked.connect(worker.request_cancel)
    worker.signals.progress.connect(on_progress)
    worker.signals.finished.connect(on_finished)
    worker.signals.cancelled.connect(on_cancelled)

    QThreadPool.globalInstance().start(worker)


