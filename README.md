# vesuvius-sheet-tools

CPU-friendly tools for **cleaning and instance-separating surface predictions** of
Herculaneum scrolls ([Vesuvius Challenge](https://scrollprize.org)). Everything
streams from the public S3 bucket — no bulk downloads, no GPU, no credentials.

Motivation: the official surface predictions are a great starting point, but they
contain false positives outside the scroll and merge into a single undifferentiated
mask. Downstream work (segmentation, labeling, virtual unwrapping) needs *clean,
per-sheet* labels. This addresses the label-quality problems described in wishlist
issues [#191](https://github.com/ScrollPrize/villa/issues/191),
[#192](https://github.com/ScrollPrize/villa/issues/192) and
[#193](https://github.com/ScrollPrize/villa/issues/193).

Tested on two scrolls with very different geometry and scan parameters:
**Scroll 3 (PHerc0332**, 2.4 µm, 78 keV**)** and **PHerc1218** (8.6 µm, 116 keV,
heavily compressed, no human segments yet).

## Quick start

```bash
pip install -r requirements.txt
cd scripts

# stream one CT slice of Scroll 3 and save it as PNG (a few MB transferred)
python quickstart_slice.py 4 z

# clean the official surface prediction of PHerc1218 (CT gating + component filter)
python clean_surface_prediction.py pherc1218

# split the cleaned surface mask into individual sheet instances
python separate_sheets.py pherc1218 1
```

Outputs (renders + `.npy` label crops) land in `output/`.

## Tools

| Script | What it does |
|---|---|
| `quickstart_slice.py` | Stream any pyramid level/slice of a scroll volume to PNG |
| `list_scroll_data.py` | List volumes/segments/predictions available for a sample in S3 |
| `overlay_surface.py` | Overlay official surface predictions on the CT (QC view) |
| `clean_surface_prediction.py` | CT-mask gating + small-component removal for prediction zarrs |
| `separate_sheets.py` | Watershed sheet-instance separation + calibrated over-segmentation merging |

All tools are multi-scroll: sample configs (volume/prediction URLs and pyramid
alignment) live in the `SCROLLS` dict in `clean_surface_prediction.py` — adding a
scroll is a 6-line entry.

## Method

**Cleaning** (`clean_surface_prediction.py`):
1. Gate predicted voxels by the masked CT (`ct > 0`): removes chunk-aligned false
   positives floating in air outside the scroll.
2. Drop 3D connected components below a voxel threshold (noise specks).

**Sheet separation** (`separate_sheets.py`):
1. Euclidean distance transform (float32) inside the cleaned mask.
2. Seed cores where distance ≥ 2 voxels (deep inside a sheet).
3. Watershed on the negated distance, constrained to the mask → sheet instances.
4. **Merge over-segmented instances**: adjacent instance pairs are merged when the
   mean depth of their shared border is ≥ 0.75× the shallower instance's interior
   depth (90th percentile) — i.e. the watershed cut was artificial. Two guards
   prevent over-merging: borders wider than `MAX_BORDER = 60` voxel pairs are
   rejected (broad contacts run *along* two stacked sheets, artificial cuts run
   *across* one sheet), and unions are capped at 15% of the mask. `MAX_BORDER` was
   calibrated empirically (see below). Union-find applies merges transitively;
   border statistics are accumulated in fixed-size chunks to bound memory.

## Results

### Cleaning (official m7 surface predictions, April 2026)

| Scroll | Predicted voxels in air (removed) | Noise components removed |
|---|---:|---:|
| Scroll 3 (PHerc0332), 64-slice crop @L2 | **65.7%** | 150,463 of 155,652 |
| PHerc1218, 64-slice crop @L3 | **50.0%** | 231,027 of 233,784 |

| Scroll 3 before | Scroll 3 after |
|---|---|
| ![before](docs/images/scroll3_clean_before.png) | ![after](docs/images/scroll3_clean_after.png) |

### Sheet separation (PHerc1218, 512×512×128 crops @L1, ~17 µm/voxel)

| Crop | Instances (watershed) | After merge | Largest instance | Merge time |
|---|---:|---:|---:|---:|
| Central (5812,1898,1898) | 491 | 391 | 4.57% of mask | 2.1 s |
| Compressed tip (5812,2320,860) | 401 | 362 | 15.55% of mask* | 1.9 s |

\* pre-existing watershed artifact, not created by merging — see Limitations.

`MAX_BORDER` calibration on the central crop (why 60):

| MAX_BORDER | Instances | Largest instance | Verdict |
|---:|---:|---:|---|
| 1000 | 180 | 14.89% | percolation (chain-merging through compressed stacks) |
| 100 | 304 | 11.84% | one large chain remains |
| 75 | 350 | 7.91% | above target |
| **60** | **391** | **4.57%** | chosen |

Each color below is one sheet instance (papyrus wrap):

| Watershed only | + calibrated merge |
|---|---|
| ![premerge](docs/images/pherc1218_sheets_premerge.png) | ![merged](docs/images/pherc1218_sheets_merged.png) |

Compressed tip (the hard case):

| Watershed only | + calibrated merge |
|---|---|
| ![premerge](docs/images/pherc1218_compressed_premerge.png) | ![merged](docs/images/pherc1218_compressed_merged.png) |

Full command-by-command validation logs: the numbers above were reproduced
end-to-end on two independent Python environments (3.11 and 3.14).

## Known limitations

- **Crushed regions**: where the scroll is badly crushed, the watershed itself
  produces a merged mega-instance (15.5% of mask in the compressed-tip crop) that
  label merging cannot split — splitting watershed output there (e.g. via local
  orientation/normal analysis) is the next work item.
- **Crop-local**: instances are labeled per crop; whole-scroll processing needs
  overlapping chunks + instance stitching (planned).
- Merging uses border salience only; orientation continuity is not yet a criterion.

## Data

All inputs stream from the Vesuvius Challenge open-data bucket
(`s3://vesuvius-challenge-open-data/`, also via HTTPS), licensed
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) by the Vesuvius
Challenge. This repository contains code only (MIT) plus small rendered
illustrations derived from that data.

## License

MIT — see [LICENSE](LICENSE).
