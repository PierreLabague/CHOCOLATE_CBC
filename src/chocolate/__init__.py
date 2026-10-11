"""Analysis pipeline for the CHOCOLATE 2 coherent beam combining experiment.

The package is organised in layers, from raw files to figures:

``chocolate.config``
    Local configuration: where the data lives, which conventions to use.
``chocolate.style``
    House graphical style shared by every figure.
``chocolate.database``
    Reading ``data_base.json`` and selecting shots by experimental
    conditions.
``chocolate.phasics_io``
    Parsing the SID4 XML exports and the raw interferograms, and writing
    the condensed ``.npy`` results.
``chocolate.zernike``
    Zernike conventions and conversions between normalisations.
``chocolate.efficiency``
    Coherent beam combining efficiency models.
``chocolate.plotting``
    Reusable plotting helpers.

A typical session only needs::

    from chocolate.config import load_config
    from chocolate.style import apply_style

    cfg = load_config()
    cfg.check()
    apply_style()
"""

__version__ = "1.0.0"
