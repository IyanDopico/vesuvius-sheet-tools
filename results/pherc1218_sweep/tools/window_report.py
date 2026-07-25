#!/usr/bin/env python3
"""Relatorio completo de uma janela do sweep PHerc1218.

Uso: python3 window_report.py <out_dir_do_run>
Emite: fracoes por seed, split rel/same, dr multi-z (centro +-200),
e a linha pronta da tabela do RESULTS.md.
"""
import json, glob, re, sys, os, collections
import numpy as np
from PIL import Image

run_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
name = os.path.basename(run_dir)
m = re.search(r'slice-(\d+)-(\d+)', name)
zb, ze = (int(m.group(1)), int(m.group(2))) if m else (None, None)

d = json.load(open(f'{run_dir}/satisfied_fitted.json'))

# --- seeds ---
seeds = []
for p in d['patches']:
    seeds.append((p['id'].replace('seed-', '').replace('-pherc1218', ''),
                  100 * p['fraction'], p['total_area']))
seeds.sort(key=lambda t: -t[1])
print('== seeds ==')
for sid, frac, area in seeds:
    print(f'  {sid}: {frac:.1f}%  (area {area:.0f})')

# --- split rel/same ---
g = collections.defaultdict(lambda: [0, 0])
for pcl in d['pcls']:
    src = pcl['source_file'].split('/')[-1]
    g[src][0] += pcl['satisfied_points']; g[src][1] += pcl['total_points']
print('== constraints ==')
split = {}
for src, (s, t) in sorted(g.items()):
    key = 'rel' if 'relative' in src else ('same' if 'same' in src else src)
    split[key] = (s, t)
    print(f'  {key}: {s}/{t} ({100*s/t:.1f}%)')

# --- dr multi-z ---
umb = json.load(open(f'{run_dir}/../../spiral_input_pherc1218/umbilicus.json'))['control_points']
uz = np.array([p['z'] for p in umb]); uy = np.array([p['y'] for p in umb]); ux = np.array([p['x'] for p in umb])
o = np.argsort(uz)
zc = (zb + ze) / 2.0 if zb else 10100.0
Z_LIST = [zc - 200, zc, zc + 200]

per_z = {z: {} for z in Z_LIST}
for wd in sorted(glob.glob(f'{run_dir}/meshes/fitted_*/w[0-9]*')):
    w = int(re.search(r'w(\d+)', wd.split('/')[-1]).group(1))
    try:
        x = np.array(Image.open(f'{wd}/x.tif'), dtype=np.float64)
        y = np.array(Image.open(f'{wd}/y.tif'), dtype=np.float64)
        z = np.array(Image.open(f'{wd}/z.tif'), dtype=np.float64)
    except Exception:
        continue
    base = (x > 0) & (y > 0) & (z > 0)
    for Z0 in Z_LIST:
        mm = base & (np.abs(z - Z0) < 30)
        if mm.sum() < 20: continue
        cy = np.interp(Z0, uz[o], uy[o]); cx = np.interp(Z0, uz[o], ux[o])
        per_z[Z0][w] = np.median(np.sqrt((x[mm]-cx)**2 + (y[mm]-cy)**2))

drs = []
print('== dr ==')
for Z0 in Z_LIST:
    radii = per_z[Z0]; ws = sorted(radii)
    gaps = np.array([radii[b]-radii[a] for a, b in zip(ws, ws[1:]) if b == a+1])
    if len(gaps) == 0:
        print(f'  z={Z0:.0f}: sem gaps'); drs.append(float('nan')); continue
    med = np.median(gaps) / 2
    drs.append(med)
    print(f'  z={Z0:.0f}: {med:.2f} L1 (IQR {np.percentile(gaps,25)/2:.2f}-{np.percentile(gaps,75)/2:.2f}, {len(gaps)} gaps)')

# --- linha da tabela ---
rel = split.get('rel', (0, 1)); same = split.get('same', (0, 1))
seeds_txt = ', '.join(f'{sid} @ {frac:.1f}%' for sid, frac, _ in seeds)
dr_txt = ' / '.join(f'{v:.2f}' for v in drs)
n_ok = sum(1 for _, frac, _ in seeds if frac >= 97)
ver = 'seeds ' + f'{n_ok}/{len(seeds)} >=97'
print('\n== linha RESULTS.md (completar data/wall/loss/veredito) ==')
print(f'| {zb}\u2013{ze} | DD/MM | WALL | {len(seeds)}: {seeds_txt} | '
      f'{100*rel[0]/rel[1]:.1f} ({rel[0]}/{rel[1]}) | {100*same[0]/same[1]:.1f} ({same[0]}/{same[1]}) | '
      f'{dr_txt} | LOSS | {ver} | |')
