# CHOCOLATE 2 — coherent beam combining analysis

Analysis pipeline for the CHOCOLATE 2 experimental campaign, carried out on the
HERA laser facility at LULI. The experiment characterises the wavefront and
piston stability of two high-energy Nd:glass amplification chains
(2 x 200 J, 90 mm beams) in view of coherent beam combining (CBC).

This repository contains the code that turns raw PHASICS SID4 acquisitions into
the figures of the manuscript. It is written so that the results can be
reproduced, inspected and reused by someone who did not run the experiment.

## What the code does

The processing is a four-stage pipeline:

1. **Acquisition** — a Phantom high-speed camera behind a PHASICS SID4
   four-wave lateral shearing interferometer (kHz-SID4)  records the interferograms of
   both beams around each laser shot.
2. **Wavefront analysis** — the SID4 software projects each frame on the
   Zernike basis up to radial order 4 and exports one XML file per frame.
3. **Partitioning** — Python condenses thousands of XML and TIFF files per
   shot into a handful of `.npy` arrays (a 1.34 GB acquisition becomes about
   130 kB). This is the only stage that needs the raw data.
4. **Interpretation** — cross-shot analysis, CBC efficiency estimation and
   figure production. This stage runs in minutes on a laptop from the
   condensed results alone.

Each measurement is identified by the time it was taken, in the form
`YYYY-MM-DD-HH-MM`, which is the key used everywhere in the code and in
`data_base.json`.

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-account>/chocolate-cbc-analysis.git
cd chocolate-cbc-analysis
```

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) creates the virtual environment, installs the
package in editable mode and pins every dependency version in `uv.lock`.

Install uv once:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from the repository root:

```bash
uv sync --extra progress
```

This creates `.venv/`, installs the `chocolate` package in editable mode with
its dependencies, the optional `tqdm` progress bars and the `dev` tools
(`black`, `ruff`). Use `uv sync --no-dev` to skip the dev tools.

Run a script inside the environment without activating it:

```bash
uv run python scripts/01_overview.py
```

or activate it as usual (`source .venv/bin/activate` on Linux / macOS,
`.venv\Scripts\activate` on Windows).

### With venv and pip

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[progress]"
pip install --group dev   # optional, requires pip >= 25.1
```

The `-e` flag installs the package in editable mode: Python records the
location of the source instead of copying it, so any change to the code takes
effect immediately.

### Configuration

Then create your local configuration:

```bash
cp config/config.example.yaml config/config.yaml
```

and edit the `root` entry so that it points to the directory containing
`data_base.json`, `PHASICS/` and `RESULTS/`.

Figures use the Open Sans font. If it is not installed, matplotlib falls back
to DejaVu Sans and everything still works.

## Data

The code and the full dataset are distributed separately.

| What | Size | Where |
|---|---|---|
| Raw acquisitions (`.tif`, `.cine`, SID4 XML) | about 1 TB | not distributed |
| Condensed results (`RESULTS/`) and `data_base.json` | about 1 GB | Zenodo, DOI to be added |
| Demonstration dataset (one shot) | a few MB | `data/demo/` in this repository |

Only the condensed results are needed to reproduce every figure of the
interpretation stage. The demonstration dataset lets you run the scripts
without downloading anything.

## Repository layout

```
config/          local configuration (config.yaml is not tracked)
src/chocolate/   the library
scripts/         one script per figure family of the manuscript
data/            demonstration dataset and pointer to the full archive
figures/         generated output, not tracked
docs/            conventions and notes
```

## Conventions

Zernike coefficients can be expressed in three normalisations of the same
basis, and the choice is always stated explicitly:

| Label | Definition | Also known as |
|---|---|---|
| `amp` | bare polynomial, max&#124;Z&#124; = 1 for every mode | SID4 "PtV mode" |
| `rms` | Noll / OSA-ANSI orthonormal basis | standard orthonormal |
| `ptv` | scaled to the true peak-to-valley excursion of the mode | optical specifications |

See `src/chocolate/zernike.py` for the exact conversion factors and the
mode ordering.

## Citing

If this code or the associated dataset is useful to you, please cite it using
the metadata in `CITATION.cff`.

## License

MIT, see `LICENSE`.
