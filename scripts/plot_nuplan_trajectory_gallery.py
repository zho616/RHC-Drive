"""Paired, unmodified nuPlan traces on vector maps, with time-gradient paths."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


# Figure contract: selected matched cases illustrate successful Context rollouts
# and baseline departures; they do not estimate aggregate generalization gains.
# Quantitative paired grid: full, unsmoothed traces and actual navigation routes.
# Python-only; shared geometry/crop per pair, calibrated maps, editable PDF/SVG.
METHODS = ('r2dreamer', 'r2dreamer_context')
TITLES = ('R2Dreamer', 'R2Dreamer + Context (Ours)')
COLORS = (('#96C5E8', '#165796'), ('#F4B3AF', '#C52C39'))


def load_case(folder, view_span=None, features_dir=None, context_margin=0.):
    result = json.loads((folder / 'results.json').read_text())
    assert result['pairing_verified']
    assert result['mass_scale'] == 1. and result['friction_scale'] == .6
    traces = []
    for method in METHODS:
        with (folder / f'{method}_trajectory.csv').open() as f:
            trace = list(csv.DictReader(f))
        traces.append(np.array([[float(r[k]) for k in ('x', 'y', 'time_s')] for r in trace]))
        assert result['methods'][method]['training_steps'] == 500000
    ref = np.loadtxt(folder / 'reference_navigation.csv', delimiter=',', skiprows=1)
    polygons = [np.asarray(p)[:, :2] for p in json.loads((folder / 'map_polygons.json').read_text())]
    # Translation only, with an equal metric aspect and identical full-route crop.
    origin = traces[0][0, :2].copy()
    ref -= origin
    for trace in traces:
        trace[:, :2] -= origin
    polygons = [p - origin for p in polygons]
    xy = np.concatenate([ref] + [t[:, :2] for t in traces])
    lo, hi = xy.min(0), xy.max(0)
    center = (lo + hi) / 2
    span = max(hi - lo) + max(16., .12 * max(hi - lo))
    span = max(span, max(hi - lo) + 2 * context_margin)
    if view_span is not None:
        # Fixed macro field of view, never tightened around a shorter trajectory.
        span = max(float(view_span), span)
    bounds = [center[0] - span/2, center[0] + span/2,
              center[1] - span/2, center[1] + span/2]
    visible = []
    region = box(bounds[0], bounds[2], bounds[1], bounds[3])
    for p in polygons:
        if len(p) >= 3:
            shape = Polygon(p)
            if not shape.is_valid:
                shape = shape.buffer(0)
            if shape.intersects(region):
                visible.append(shape)
    roads = unary_union(visible)
    features = None
    if features_dir is not None:
        scene = result['methods'][METHODS[0]]['scene']
        features = json.loads((features_dir / f'scene_{scene}_context.json').read_text())
        assert features['scenario_id'] == result['methods'][METHODS[0]]['scenario_id']
        for line in features['lines']:
            line['xy'] = (np.asarray(line['xy'])[:, :2] - origin).tolist()
        for vehicle in features['vehicles']:
            vehicle['xy'] = (np.asarray(vehicle['xy']) - origin).tolist()
    return dict(folder=folder, result=result, traces=traces, ref=ref,
                roads=roads, polygons=polygons, bounds=bounds, span=span,
                features=features)


def draw_panel(ax, case, index, minimal=False):
    ref, trace = case['ref'], case['traces'][index]
    lo_x, hi_x, lo_y, hi_y = case['bounds']
    shapes = list(case['roads'].geoms) if hasattr(case['roads'], 'geoms') else [case['roads']]
    for road in shapes:
        if road.geom_type != 'Polygon':
            continue
        ax.fill(*np.asarray(road.exterior.coords).T, facecolor='#FAFAFA',
                edgecolor='#949A9E', linewidth=.65, zorder=0)
        for hole in road.interiors:
            ax.fill(*np.asarray(hole.coords).T, facecolor='white',
                    edgecolor='#949A9E', linewidth=.65, zorder=0)
    # Thin internal lane boundaries; exterior union above removes polygon seams.
    if case['features'] is None:
        ax.add_collection(PolyCollection(case['polygons'], facecolors='none',
                          edgecolors='#CDD0D2', linewidths=.25, zorder=1))
    else:
        for line in case['features']['lines']:
            broken = 'BROKEN' in line['type']
            edge = 'EDGE' in line['type']
            ax.plot(*np.asarray(line['xy']).T, color='#8F969B' if edge else '#B7BDC1',
                    linewidth=.55 if edge else .4,
                    linestyle=(0, (3, 3)) if broken else '-', zorder=1)
        # Shared initial traffic context, with trajectories shown above it.
        for vehicle in case['features']['vehicles']:
            heading = vehicle['heading']
            rotation = np.array([[np.cos(heading), -np.sin(heading)],
                                 [np.sin(heading), np.cos(heading)]])
            length, width = vehicle['length'], vehicle['width']
            corners = np.array([[-length/2,-width/2],[length/2,-width/2],
                                [length/2,width/2],[-length/2,width/2]])
            corners = corners @ rotation.T + vehicle['xy']
            ax.fill(*corners.T, color='#CBD0D4', ec='#939BA1', lw=.35, zorder=2)
    ax.plot(*ref.T, color='#E6A23C', linewidth=1.15, linestyle=(0, (4, 2.5)), zorder=3)
    xy = trace[:, :2]
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list(f'method_{index}', COLORS[index])
    # A common simulation-time normalization within each case preserves timing.
    norm = Normalize(0., max(t[-1, 2] for t in case['traces']))
    ax.add_collection(LineCollection(segments, colors='white', linewidths=3.4,
                                    capstyle='round', zorder=4))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.25,
                        capstyle='round', zorder=5)
    lc.set_array(trace[1:, 2])
    ax.add_collection(lc)
    ax.scatter(*xy[0], s=17, facecolor='white', edgecolor=COLORS[index][1],
               linewidth=.9, zorder=6)
    # Final vehicle footprint uses actual XL dimensions and the last motion heading.
    delta = xy[-1] - xy[max(0, len(xy)-4)]
    heading = np.arctan2(delta[1], delta[0])
    rot = np.array([[np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)]])
    body = np.array([[-2.87, -1.15], [2.87, -1.15], [2.87, 1.15], [-2.87, 1.15]])
    body = body @ rot.T + xy[-1]
    ax.fill(*body.T, facecolor=cmap(norm(trace[-1, 2])), edgecolor='white',
            linewidth=.6, zorder=7)
    ax.set(xlim=(lo_x, hi_x), ylim=(lo_y, hi_y), aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color('#D0D2D4')
        spine.set_linewidth(.65)
    if minimal:
        return
    # Calibrated distance, kept in the same location in paired panels.
    scale = 20 if case['span'] >= 100 else 10 if case['span'] >= 50 else 5
    x, y = lo_x + .065*case['span'], lo_y + .065*case['span']
    ax.plot([x, x+scale], [y,y], color='#454A4E', lw=1.3, zorder=9)
    ax.text(x+scale/2, y+.025*case['span'], f'{scale} m', ha='center', va='bottom',
            fontsize=6.5, color='#454A4E', zorder=9)
    info = case['result']['methods'][METHODS[index]]
    outcome = 'Success' if info['success'] else 'Collision' if info['crash'] else 'Out of road' if info['out_of_road'] else 'Timeout'
    ax.text(.5, -.045, f'{outcome}  |  Route completion {info["route_completion"]:.1%}',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
            color=COLORS[index][1])


def save_figure(cases, target, minimal=False):
    n = len(cases)
    # Up to four paired cases in a 2 x 4 gallery, keeping each map square.
    pair_cols = 3 if n == 3 else 2 if n > 2 else 1
    rows = int(np.ceil(n / pair_cols))
    width = 18.4 if pair_cols == 3 else 12.4 if pair_cols == 2 else 6.5
    height = rows * 3.35 + (.12 if minimal else .65)
    fig = plt.figure(figsize=(width, height), facecolor='white')
    grid = fig.add_gridspec(rows, pair_cols*2, left=.022, right=.986,
                           bottom=.015 if minimal else .07, top=.94,
                           wspace=.13, hspace=.17 if minimal else .30)
    for i, case in enumerate(cases):
        row, pair = divmod(i, pair_cols)
        for j in range(2):
            ax = fig.add_subplot(grid[row, pair*2+j])
            draw_panel(ax, case, j, minimal=minimal)
            ax.set_title(TITLES[j], fontsize=10, pad=7,
                         color=COLORS[j][1], fontweight='medium')
            if j == 0:
                scene = case['result']['methods'][METHODS[0]]['scene']
                split = Path(case['result'].get('scenario_directory', 'test')).name
                caption = f'Train scene {scene}' if split == 'train' else f'Validation scene {scene}' if split == 'val' else f'Scene {scene}'
                ax.text(.035, .965, f'({chr(97+i)})  {caption}', transform=ax.transAxes,
                        fontsize=9, fontweight='bold', ha='left', va='top', zorder=10,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
    # Shared reference and progression keys; pale -> dark means early -> late.
    if not minimal:
        fig.legend([Line2D([], [], color='#E6A23C', lw=1.4, ls=(0, (4, 2.5))),
                Line2D([], [], marker='o', mfc='white', mec='#555A60', ls='none', markersize=4)],
               ['Reference navigation', 'Start'], loc='lower left', bbox_to_anchor=(.022, .001),
               frameon=False, ncol=2, fontsize=8, handlelength=2.7)
    for j, x in ([] if minimal else enumerate([.50, .75])):
        bar = fig.add_axes([x, .026, .10, .012])
        bar.imshow(np.linspace(0, 1, 256)[None], aspect='auto',
                   cmap=LinearSegmentedColormap.from_list('key', COLORS[j]))
        bar.axis('off')
        fig.text(x+.105, .031, 'Early → Late', fontsize=7, va='center', color='#555A60')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(target.with_suffix('.'+ext), dpi=350, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--view-span', type=float,
                        help='Minimum square world view in metres, shared by all panels')
    parser.add_argument('--minimal', action='store_true',
                        help='Remove scale bars, outcome captions, and footer legend')
    parser.add_argument('--context-margin', type=float, default=0.,
                        help='Minimum context in metres on each side of complete paths')
    parser.add_argument('--features-dir', type=Path,
                        help='Optional processed scenario road lines and initial traffic context')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    cases = [load_case(p.resolve(), args.view_span, args.features_dir,
                       args.context_margin) for p in args.cases]
    save_figure(cases, args.output/'trajectory_gallery', minimal=args.minimal)
    for c in cases:
        scene = c['result']['methods'][METHODS[0]]['scene']
        save_figure([c], args.output/f'scene_{scene}_paired', minimal=args.minimal)
    provenance = dict(cases=[dict(scene=c['result']['methods'][METHODS[0]]['scene'],
                                  source=str(c['folder']), results=c['result']) for c in cases],
                      transformations='Translation only; equal metric aspect; shared square full-route bounds per pair.',
                      trajectories='Full recorded trajectories; no smoothing, interpolation, or outcome editing.',
                      gradients='Blue baseline, red Context; pale to dark is time with shared normalization per pair.',
                      vehicle_glyph='XL 5.74 x 2.30 m; final orientation estimated from last three motion steps.',
                      selection='Post-hoc favorable examples. One matched inference seed per scene; not aggregate evaluation.')
    provenance.update(view_span_m=args.view_span, minimal=args.minimal,
                      context_margin_m=args.context_margin,
                      map_context='Processed original road boundaries and lane markings; shared initial traffic' if args.features_dir else 'Lane polygons',
                      panel_bounds=[c['bounds'] for c in cases])
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(args.output/'trajectory_gallery.png')


if __name__ == '__main__':
    main()
