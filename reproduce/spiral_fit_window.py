"""Reproduce the PHerc1218 spiral fit on one z window (stable PCL-only config).

Downloads the villa spiral scripts at a pinned commit of the IyanDopico/villa
fork (upstream + the atlas-lookup fix of ScrollPrize/villa#1207), the
spiral_input_pherc1218 pack from this repo, patches fit_spiral.py's hardcoded
config header for PHerc1218, and runs the fit. See REPRODUCING.md for the
expected satisfaction band and caveats.

Works on any machine with a CUDA GPU (>= ~6 GB for an 800-slice window),
Python >= 3.11 and internet access. No Kaggle assumptions.

Environment overrides (all optional):
  FIT_Z_BEGIN / FIT_Z_END   window in full-resolution voxels (default 9700/10500)
  FIT_STEPS                 training steps (default 30000)
  FIT_SEED                  random seed (default 1)
  FIT_WORK                  working directory (default ./spiral_work)

Deps: torch (CUDA), numpy, scipy, pillow, tqdm, einops, kornia, trimesh,
pyro-ppl, torchdiffeq, wandb, zarr (2.x; pin numcodecs<0.16 with zarr 2.18).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

T0 = time.time()

VILLA_COMMIT = "61bd95c75e91b082f8de6964f5edbc5bc6a54eb7"  # fork: HEAD + #1207 fix
PACK_REF = "main"
REPO_RAW = "https://raw.githubusercontent.com/IyanDopico/vesuvius-sheet-tools"
VILLA_RAW = ("https://raw.githubusercontent.com/IyanDopico/villa/"
             f"{VILLA_COMMIT}/volume-cartographer/scripts/spiral")

Z_BEGIN = int(os.environ.get("FIT_Z_BEGIN", "9700"))
Z_END = int(os.environ.get("FIT_Z_END", "10500"))
STEPS = int(os.environ.get("FIT_STEPS", "30000"))
SEED = int(os.environ.get("FIT_SEED", "1"))
WORK = os.path.abspath(os.environ.get("FIT_WORK", "./spiral_work"))
SPIRAL_SENSE = "CW"   # measured on slab z4928; see REPRODUCING.md caveat

SPIRAL_DIR = f"{WORK}/spiral"
DATASET_DIR = f"{WORK}/spiral_input_pherc1218"
OUT_DIR = f"{WORK}/out"

CONFIG_OVERRIDES = {
    "random_seed": SEED,
    "num_training_steps": STEPS,
    "loss_weight_dense_normals": 0.0,   # PCL-only: no lasagna inputs
    "loss_weight_dense_spacing": 0.0,
    "loss_weight_shell_outer": 0.0,     # no outer shell for this scroll
    "erode_patches": 0,                 # keep small seed patches alive
    "shell_outer_winding_idx": None,    # no winding cap on saved meshes
    "output_first_winding": 0,
    "initial_dr_per_winding": 20.0,     # measured pitch 173 um / 8.64 um
    "gap_expander_num_windings": 130,
    # CRITICAL for parity with the published runs: per-step sample counts are
    # scaled by (z_end-z_begin)/9500 at runtime; the default 84 becomes ~7
    # unattached strips/step on an 800-slice window and satisfaction lands
    # ~9 pts low (found via pscamillo's independent reproduction, 2026-07-23).
    # 800 scales to ~67/step, matching the published band.
    "unattached_pcl_num_per_step": 800,
    "stratified_pcl_sampling": False,   # pin-parity with the published runs
    "dt_target_mode": "strip_median",
    "save_png_visualizations": True,
}

VILLA_FILES = [
    "fit_spiral.py", "spiral_helpers.py", "losses.py", "point_collection.py",
    "tifxyz.py", "umbilicus.py", "tracks.py", "transforms.py", "geom_utils.py",
    "ddp_helpers.py", "lasagna_data.py", "flow_fields.py", "sample_spiral.py",
    "satisfaction_metrics.py", "visualization.py", "checkpoint_io.py",
    "influence.py", "native_spiral.py", "dt_targets.py", "loss_maps.py",
    "sdt_losses.py", "soft_alignment.py", "prefetch.py", "strip_path_pools.py",
    "strip_paths.py", "gap_triton.py", "flow_triton.py", "lasagna_mmap.py",
    "geometry_snapshot.py",
]
PACK_FILES = [
    "umbilicus.json", "same_windings.json", "relative_windings.json",
    "abs_winding.json", "README.txt",
]

HEADER = f"""
# PHerc1218 (volume 20250521120456, 8.64 um full-res, grid 23247x7593x7593 zyx)
dataset_path = {DATASET_DIR!r}
scroll_zarr_path = None
normal_nx_zarr_path = None
normal_ny_zarr_path = None
grad_mag_zarr_path = None
normal_zarr_group = '2'
pcl_json_paths = [
    f'{{dataset_path}}/abs_winding.json',
    f'{{dataset_path}}/relative_windings.json',
    f'{{dataset_path}}/same_windings.json',
]
fibers_path = None
verified_patches_path = f'{{dataset_path}}/verified_patches'
unverified_patches_path = None
run_tag = os.environ.get('FIT_SPIRAL_RUN_TAG')
shell_path = None
tracks_dbm_path = None
spiral_outward_sense = {SPIRAL_SENSE!r}
umbilicus_z_to_yx = lambda: json_umbilicus_z_to_yx(f'{{dataset_path}}/umbilicus.json', coordinate_scale=1.0)
scroll_name = 'pherc1218'
z_begin, z_end = {Z_BEGIN}, {Z_END}
voxel_size_um = 8.64
cache_path = os.environ.get('FIT_SPIRAL_CACHE_DIR', {WORK + '/cache'!r})
lasagna_scale = 2
render_volume_scale = int(os.environ.get('FIT_SPIRAL_RENDER_VOLUME_SCALE', '16'))
pcl_input_specs = None
lasagna_storage_backend = 'auto'
surf_sdt_zarr_path = None
surf_sdt_zarr_group = '1'

"""


def fetch(url, dst, attempts=5):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"retry {i + 1} for {url}: {e}", flush=True)
            time.sleep(2 ** i)


def main():
    print(f"[{time.time() - T0:6.1f}s] fetching villa spiral @ {VILLA_COMMIT[:9]}",
          flush=True)
    for f in VILLA_FILES:
        fetch(f"{VILLA_RAW}/{f}", f"{SPIRAL_DIR}/{f}")

    print(f"[{time.time() - T0:6.1f}s] fetching input pack @ {PACK_REF}", flush=True)
    base = f"{REPO_RAW}/{PACK_REF}/data/spiral_input_pherc1218"
    for f in PACK_FILES:
        fetch(f"{base}/{f}", f"{DATASET_DIR}/{f}")
    listing = json.loads(urllib.request.urlopen(
        "https://api.github.com/repos/IyanDopico/vesuvius-sheet-tools/contents/"
        f"data/spiral_input_pherc1218/verified_patches?ref={PACK_REF}").read())
    patches = [e["name"] for e in listing if e["type"] == "dir"]
    for name in patches:
        for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
            fetch(f"{base}/verified_patches/{name}/{f}",
                  f"{DATASET_DIR}/verified_patches/{name}/{f}")
    print(f"verified patches fetched: {len(patches)}", flush=True)
    assert patches, "no verified patches in the pack - fit_spiral requires >= 1"

    print(f"[{time.time() - T0:6.1f}s] patching fit_spiral.py header", flush=True)
    src_path = f"{SPIRAL_DIR}/fit_spiral.py"
    lines = open(src_path, encoding="utf-8").read().splitlines(keepends=True)
    i_hdr = next(i for i, l in enumerate(lines) if l.startswith("# PHercParis4"))
    i_cfg = next(i for i, l in enumerate(lines) if l.startswith("default_config"))
    open(src_path, "w", encoding="utf-8").write(
        "".join(lines[:i_hdr] + [HEADER] + lines[i_cfg:]))

    env = dict(os.environ)
    env.update({
        "WANDB_MODE": "disabled",
        "FIT_SPIRAL_OUT_DIR": OUT_DIR,
        "FIT_SPIRAL_CACHE_DIR": f"{WORK}/cache",
        "FIT_SPIRAL_RUN_TAG": f"repro-z{Z_BEGIN}-{Z_END}-s{SEED}",
        "FIT_SPIRAL_CONFIG_OVERRIDES": json.dumps(CONFIG_OVERRIDES),
    })
    print(f"[{time.time() - T0:6.1f}s] running fit_spiral "
          f"(z {Z_BEGIN}-{Z_END}, sense {SPIRAL_SENSE}, {STEPS} steps, "
          f"seed {SEED})", flush=True)
    log = open(f"{WORK}/fit_run.log", "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "fit_spiral.py"], cwd=SPIRAL_DIR,
                            env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    proc.wait()
    log.close()
    print(f"[{time.time() - T0:6.1f}s] exit {proc.returncode}; outputs in "
          f"{OUT_DIR} (satisfied_fitted.json is the headline)", flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
