"""Reading ``data_base.json`` and selecting shots.

Every measurement of the campaign is identified by the instant it was
taken, in the form ``YYYY-MM-DD-HH-MM``. That string is the key used
throughout the code, in ``data_base.json`` and in the names of the files of
the ``RESULTS`` directory.

The database is a flat dictionary mapping each identifier to the metadata
of the corresponding acquisition: frame rate, exposure, laser
configuration, export parameters, and so on. One reserved key, ``__meta``,
describes the database itself and is never a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

__all__ = [
    "META_KEY",
    "load_database",
    "list_measurement_ids",
    "select_ids",
    "shot_date",
    "shot_time",
]

#: Reserved key of ``data_base.json`` holding metadata about the database.
META_KEY = "__meta"


def load_database(root) -> Dict[str, dict]:
    """Read ``data_base.json`` from the data directory.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing ``data_base.json``.

    Returns
    -------
    dict
        The database as stored, including the ``__meta`` entry.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the path that was tried.
    """
    path = Path(root) / "data_base.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"data_base.json not found at {path}\n"
            f"Check the 'root' entry of your configuration."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_measurement_ids(database: Dict[str, dict]) -> List[str]:
    """Return every measurement identifier, sorted chronologically.

    The identifiers are ``YYYY-MM-DD-HH-MM`` strings, so sorting them
    alphabetically sorts them in time.
    """
    return sorted(key for key in database if key != META_KEY)


def select_ids(
    database: Dict[str, dict],
    conditions: Optional[Iterable[Callable[[str, dict], bool]]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """Select the measurements satisfying every condition.

    Parameters
    ----------
    database : dict
        As returned by :func:`load_database`.
    conditions : iterable of callable, optional
        Each callable takes ``(shot_id, metadata)`` and returns a boolean.
        A shot is kept when all of them return True. No condition means
        every shot is kept.
    exclude : iterable of str, optional
        Identifiers to drop from the result whatever the conditions say.

    Returns
    -------
    list of str
        Matching identifiers, in chronological order.

    Examples
    --------
    Shots of a given day, taken with both beams::

        select_ids(db, [
            lambda sid, meta: sid.startswith("2025-12-02"),
            lambda sid, meta: meta.get("config") == "AB",
        ])

    Notes
    -----
    A condition that raises on a given shot, typically because a field is
    missing from its metadata, discards that shot silently. This is
    deliberate: the database was filled during the campaign and some early
    entries are incomplete.
    """
    conditions = list(conditions or [])
    excluded = set(exclude or [])

    selected = []
    for shot_id in list_measurement_ids(database):
        if shot_id in excluded:
            continue
        try:
            if all(condition(shot_id, database[shot_id]) for condition in conditions):
                selected.append(shot_id)
        except Exception:
            continue
    return selected


def shot_date(shot_id: str) -> str:
    """Return the ``YYYY-MM-DD`` part of an identifier."""
    return shot_id[:10]


def shot_time(shot_id: str) -> str:
    """Return the ``HH:MM`` part of an identifier, ready for display."""
    return shot_id[11:13] + ":" + shot_id[14:16]
