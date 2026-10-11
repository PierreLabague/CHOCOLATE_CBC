"""Loading of the local configuration.

No absolute path should ever appear in the source code or be committed to
the repository. Everything machine-specific lives in ``config/config.yaml``,
which is deliberately excluded from version control. A template with dummy
values is provided as ``config/config.example.yaml``.

Usage in an analysis script::

    from chocolate.config import load_config

    cfg = load_config()
    root = cfg.root                 # pathlib.Path to the data directory
    zernike_mode = cfg.zernike_mode
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = ["Config", "load_config", "find_config_file"]

# Directory that contains this file: <repo>/src/chocolate
_PACKAGE_DIR = Path(__file__).resolve().parent
# Repository root, two levels up: <repo>
_REPO_ROOT = _PACKAGE_DIR.parent.parent


@dataclass
class Config:
    """Runtime configuration of an analysis session.

    Attributes
    ----------
    root : pathlib.Path
        Directory holding ``data_base.json``, ``PHASICS/`` and ``RESULTS/``.
    figures_dir : pathlib.Path
        Where generated figures are written.
    zernike_mode : str
        Normalisation used to express Zernike coefficients, one of
        ``'rms'``, ``'amp'`` or ``'ptv'``. See :mod:`chocolate.zernike`.
    time_ref_ms : float
        Reference time used to zero every temporal trace, in milliseconds
        relative to the laser shot. Negative means before the shot.
    coef_tilt_mrad_to_ptv : float
        Geometric factor converting a tilt expressed in mrad by the SID4
        into a peak-to-valley optical path difference in waves.
    wavelength_nm : float
        Probe wavelength of the wavefront sensor.
    extra : dict
        Any additional key found in the YAML file, kept as-is.
    """

    root: Path
    figures_dir: Path = field(default_factory=lambda: _REPO_ROOT / "figures")
    zernike_mode: str = "rms"
    time_ref_ms: float = -0.5
    coef_tilt_mrad_to_ptv: float = 3.47
    wavelength_nm: float = 1064.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.root = Path(self.root).expanduser()
        self.figures_dir = Path(self.figures_dir).expanduser()
        if self.zernike_mode not in ("rms", "amp", "ptv"):
            raise ValueError(
                f"zernike_mode must be 'rms', 'amp' or 'ptv', got "
                f"{self.zernike_mode!r}"
            )

    @property
    def database_path(self) -> Path:
        """Full path to ``data_base.json``."""
        return self.root / "data_base.json"

    def check(self) -> None:
        """Raise a readable error if the configured data directory is wrong.

        Call this at the start of a script so that a mistyped path fails
        immediately with an explicit message instead of much later with an
        obscure one.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {self.root}\n"
                f"Edit the 'root' entry of your config/config.yaml."
            )
        if not self.database_path.is_file():
            raise FileNotFoundError(
                f"data_base.json not found in {self.root}\n"
                f"The 'root' entry must point to the directory that "
                f"contains data_base.json, PHASICS/ and RESULTS/."
            )
        self.figures_dir.mkdir(parents=True, exist_ok=True)


def find_config_file() -> Path:
    """Locate the configuration file to use.

    Resolution order:

    1. the ``CHOCOLATE_CONFIG`` environment variable, if set;
    2. ``config/config.yaml`` at the repository root;
    3. ``config/config.example.yaml`` as a last resort, which will fail the
       :meth:`Config.check` call and tell the user what to do.
    """
    env = os.environ.get("CHOCOLATE_CONFIG")
    if env:
        return Path(env).expanduser()

    local = _REPO_ROOT / "config" / "config.yaml"
    if local.is_file():
        return local

    example = _REPO_ROOT / "config" / "config.example.yaml"
    if example.is_file():
        return example

    raise FileNotFoundError(
        "No configuration file found. Copy config/config.example.yaml to "
        "config/config.yaml and set the 'root' entry to your data directory."
    )


def load_config(path=None) -> Config:
    """Read the YAML configuration and return a :class:`Config`.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Explicit configuration file. Defaults to :func:`find_config_file`.
    """
    path = Path(path) if path is not None else find_config_file()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    known = {
        "root",
        "figures_dir",
        "zernike_mode",
        "time_ref_ms",
        "coef_tilt_mrad_to_ptv",
        "wavelength_nm",
    }
    kwargs = {key: raw[key] for key in known if key in raw}
    kwargs["extra"] = {key: value for key, value in raw.items() if key not in known}

    if "root" not in kwargs:
        raise KeyError(f"The configuration file {path} has no 'root' entry.")

    return Config(**kwargs)
