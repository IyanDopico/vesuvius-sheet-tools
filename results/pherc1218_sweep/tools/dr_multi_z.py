#!/usr/bin/env python3
"""dr por winding em multiplos z centrais, para um run do fit PHerc1218.

Uso: python3 dr_multi_z.py <out_dir_do_run>
Mede dr mediano entre windings consecutivos em z = 9900, 10100, 10300,
contra o umbilicus interpolado do pack. Criterio: 10.1 +- 0.8 L1 vox.
"""
import json, glob, re, sys
import numpy as np
from PIL import Image

run_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
# Z_LIST derivada do nome do diretorio (padrao ..._slice-ZB-ZE_...):
# centro e centro +-200. Override manual: passar z's como args extras.
import os as _os
_m = re.search(r'slice-(\d+)-(\d+)', _os.path.basename(_os.path.abspath(run_dir)))
if len(sys.argv) > 2:
    Z_LIST = [float(a) for a in sys.argv[2:]]
elif _m:
    _zc = (int(_m.group(1)) + int(_m.group(2))) / 2.0
    Z_LIST = [_zc - 200.0, _zc, _zc + 200.0]
else:
    Z_LIST = [9900.0, 10100.0, 10300.0]  # fallback: janela publicada
print(f'Z_LIST: {Z_LIST}')

umb = json.load(open(f'{run_dir}/../../spiral_input_pherc1218/umbilicus.json'))['control_points']
uz = np.array([p['z'] for p in umb]); uy = np.array([p['y'] for p in umb]); ux = np.array([p['x'] for p in umb])
o = np.argsort(uz)

# carrega cada winding uma vez; mede nos tres z na mesma passada
per_z_radii = {z: {} for z in Z_LIST}
dirs = sorted(glob.glob(f'{run_dir}/meshes/fitted_*/w[0-9]*'))
for d in dirs:
    w = int(re.search(r'w(\d+)', d.split('/')[-1]).group(1))
    try:
        x = np.array(Image.open(f'{d}/x.tif'), dtype=np.float64)
        y = np.array(Image.open(f'{d}/y.tif'), dtype=np.float64)
        z = np.array(Image.open(f'{d}/z.tif'), dtype=np.float64)
    except Exception:
        continue
    base = (x > 0) & (y > 0) & (z > 0)
    for Z0 in Z_LIST:
        m = base & (np.abs(z - Z0) < 30)
        if m.sum() < 20:
            continue
        cy = np.interp(Z0, uz[o], uy[o]); cx = np.interp(Z0, uz[o], ux[o])
        per_z_radii[Z0][w] = np.median(np.sqrt((x[m]-cx)**2 + (y[m]-cy)**2))

print(f'run: {run_dir}')
meds = []
for Z0 in Z_LIST:
    radii = per_z_radii[Z0]
    ws = sorted(radii)
    gaps = np.array([radii[b]-radii[a] for a, b in zip(ws, ws[1:]) if b == a+1])
    if len(gaps) == 0:
        print(f'  z={Z0:.0f}: sem gaps mensuraveis'); continue
    med = np.median(gaps)
    meds.append(med/2)
    print(f'  z={Z0:.0f}: {len(ws)} windings (w{ws[0]}..w{ws[-1]}), {len(gaps)} gaps | '
          f'dr {med:.2f} full-res = {med/2:.2f} L1 | IQR {np.percentile(gaps,25):.2f}-{np.percentile(gaps,75):.2f}')
if meds:
    print(f'  resumo: dr L1 por z = {", ".join(f"{m:.2f}" for m in meds)} | spread {max(meds)-min(meds):.2f} | criterio 10.1 +- 0.8')
