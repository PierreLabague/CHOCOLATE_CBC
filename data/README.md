# Data

The full dataset is not stored in this repository.

## Demonstration dataset

`data/demo/` contains a single measurement, complete enough to run every
script of the interpretation stage. It is tracked by git so that the
repository works out of the box.

## Full dataset

The condensed results of the whole campaign (the `RESULTS/` directory and
`data_base.json`, about 1 GB) are archived on Zenodo:

> DOI: to be added on publication

Download the archive, unpack it anywhere on your machine, and point the
`root` entry of `config/config.yaml` to the unpacked directory.

## Raw data

The raw acquisitions (about 1 TB of `.tif` interferograms and SID4 XML
exports) are not distributed. They are only needed to re-run the
partitioning stage, `scripts/00_preprocess.py`. Contact the author if you
need access to them.

## Expected directory layout

```
<root>/
├── data_base.json
├── PHASICS/
│   └── YYYY-MM-DD-HH-MM_PHASICS/
│       └── Test1/28346_1_NNN/ImgXXXXXX.tif
├── PHASICS_NEW/
│   └── YYYY-MM-DD-HH-MM_PHASICS/
│       └── SID4/YYYY-MM-DD-HH-MM_WA/Results/ACC ImgXXXXXX.xml
└── RESULTS/
    └── YYYY-MM-DD-HH-MM/
        ├── YYYY-MM-DD-HH-MM_data__WA.npy
        ├── YYYY-MM-DD-HH-MM_tag__WA.npy
        ├── YYYY-MM-DD-HH-MM_time__WA.npy
        ├── YYYY-MM-DD-HH-MM_piston.npy
        ├── YYYY-MM-DD-HH-MM_IA.npy
        └── ...
```
