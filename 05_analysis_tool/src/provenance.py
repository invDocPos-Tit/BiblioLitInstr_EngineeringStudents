"""Procedencia y trazabilidad determinista de la corrida.

Responsabilidad unica: registrar QUE se ejecuto, CON QUE corpus/entorno y
CON QUE resultado, y hacer cumplir el contrato agnostico entre revisiones
vinculando data/enriched_articles.json al SHA-256 de consolidado.csv.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import DATA_DIR, LOGS_DIR
from src.utils import load_json, save_json

logger = logging.getLogger("bibliometrico.provenance")

FINGERPRINT_PATH = DATA_DIR / "review_fingerprint.json"


def sha256_file(path: Path) -> Optional[str]:
    """SHA-256 de un archivo (integridad/citabilidad de la entrada)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _pip_freeze() -> list:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze",
             "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0:
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("pip freeze fallo: %s", e)
    return []


def environment_snapshot() -> dict:
    """Snapshot reproducible del entorno de ejecucion."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": _pip_freeze(),
    }


def current_fingerprint(csv_path: Path) -> dict:
    return {
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(Path(csv_path)),
    }


def write_fingerprint(csv_path: Path) -> None:
    """Liga el estado derivado actual al corpus (consolidado.csv) de ESTA revision."""
    fp = current_fingerprint(Path(csv_path))
    fp["written_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        save_json(fp, FINGERPRINT_PATH)
    except OSError as e:
        logger.warning("No se pudo escribir la huella de revision: %s", e)


def verify_fingerprint(csv_path: Path) -> tuple:
    """Compara la huella guardada vs. el corpus actual.

    Returns (status, detail):
      - "absent":   sin huella (checkpoint previo) -> auto-sellar
      - "match":    mismo corpus -> continuar
      - "mismatch": OTRO corpus -> abortar y pedir --new-review
    """
    saved = load_json(FINGERPRINT_PATH)
    if not isinstance(saved, dict) or not saved:
        return "absent", "sin huella previa"
    cur = current_fingerprint(Path(csv_path))
    if saved.get("csv_sha256") != cur["csv_sha256"]:
        return "mismatch", "corpus consolidado.csv cambio"
    return "match", "mismo corpus"


class RunRecorder:
    """Acumula procedencia y escribe output/logs/run_manifest_<ts>.json."""

    def __init__(self, args: Any):
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.t_start = datetime.now()
        try:
            self.args = dict(vars(args))
        except TypeError:
            self.args = {}
        self.phases_requested = list(self.args.get("phases", []) or [])
        self.phase_status: dict = {}
        self._order: list = []

    def _set(self, name: str, status: str, detail: str = "") -> None:
        if name not in self.phase_status:
            self._order.append(name)
        self.phase_status[name] = {"status": status, "detail": detail}

    @contextlib.contextmanager
    def phase(self, name: str, swallow: bool = True):
        """Envuelve una fase registrando su estado de ejecucion."""
        self._set(name, "running")
        try:
            yield
        except ImportError as e:
            self._set(name, "skipped", f"ImportError: {e}")
            logger.warning("%s omitida (dependencia ausente): %s", name, e)
            if not swallow:
                raise
        except SystemExit:
            self._set(name, "exit", "sys.exit")
            raise
        except Exception as e:
            self._set(name, "failed", f"{type(e).__name__}: {e}")
            logger.error("%s fallo: %s", name, e, exc_info=True)
            if not swallow:
                raise
        else:
            self._set(name, "ok", "")

    def finalize(self, articles: Any, csv_path: Path,
                 extra: Optional[dict] = None) -> Optional[Path]:
        t_end = datetime.now()
        fp = current_fingerprint(Path(csv_path))
        manifest = {
            "run_id": self.ts,
            "timestamp_start": self.t_start.isoformat(timespec="seconds"),
            "timestamp_end": t_end.isoformat(timespec="seconds"),
            "duration_s": round((t_end - self.t_start).total_seconds(), 1),
            "args": self.args,
            "phases_requested": self.phases_requested,
            "phases_executed": [n for n in self._order
                                if self.phase_status[n]["status"] == "ok"],
            "phase_status": {n: self.phase_status[n] for n in self._order},
            "inputs": {
                "csv_path": fp["csv_path"],
                "csv_sha256": fp["csv_sha256"],
                "n_articles": (len(articles)
                               if isinstance(articles, list) else None),
            },
            "environment": environment_snapshot(),
        }
        if extra:
            manifest.update(extra)
        path = LOGS_DIR / f"run_manifest_{self.ts}.json"
        try:
            save_json(manifest, path)
            logger.info("Manifiesto de corrida: %s", path)
        except OSError as e:
            logger.warning("No se pudo escribir el manifiesto: %s", e)
            return None
        return path
