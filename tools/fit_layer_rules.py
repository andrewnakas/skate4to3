#!/usr/bin/env python3
"""Choose, per shader, the colour layer and the tint rule that best reproduce skate.'s own baked
proxy colours (work/layer_samples.json from tools/calibrate_layers.py), by 2-fold cross-validation.

    SKATE_WORK=work python tools/fit_layer_rules.py   -> work/layer_choice.json

A rule is kept only where it beats the default (first colour layer by base-slot rank, no tint) on
held-out sections of that shader by >= 10% and the shader has >= 8 sections.
"""
import collections, json, os, sys
import numpy as np

WORK = os.environ.get('SKATE_WORK', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work'))
EPS = 1e-3


def err(pred, want):
    return float(np.abs(np.log((np.asarray(pred) + EPS) / (np.asarray(want) + EPS))).mean())


def predict(s, layer, tint):
    lay = s['layers'].get(layer) if layer else next(iter(s['layers'].values()))
    if lay is None:
        return None
    m = np.array(lay[1])
    if tint is None:
        return m
    h, mode = tint
    v = s['vec3'].get(h)
    if v is None:
        return m
    v = np.array(v)
    return {'mul': m * v, 'mul2': m * 2 * v, 'replace': np.clip(v, 0, None) ** 2.2}[mode]


def score(samples, layer, tint):
    e = [err(p, s['want']) for s in samples if (p := predict(s, layer, tint)) is not None]
    return np.mean(e) if e else 9.0


def main():
    S = json.load(open(os.path.join(WORK, 'layer_samples.json')))
    by = collections.defaultdict(list)
    for s in S:
        by[s['shader']].append(s)
    out, gain = {}, []
    for sh, ss in by.items():
        if len(ss) < 8:
            continue
        layers = sorted({h for s in ss for h in s['layers']})
        tints = [None] + sorted({(h, m) for s in ss for h, v in s['vec3'].items()
                                 if not np.allclose(v, 0.5) for m in ('mul', 'mul2', 'replace')})
        cands = [(l, t) for l in [None] + layers for t in tints]
        folds = [ss[0::2], ss[1::2]]
        held = []
        for k in (0, 1):
            fit = min(cands, key=lambda c: score(folds[k], *c))
            held.append((score(folds[1 - k], *fit), score(folds[1 - k], None, None)))
        rule_err = np.mean([h[0] for h in held]); base_err = np.mean([h[1] for h in held])
        if rule_err <= 0.9 * base_err:
            best = min(cands, key=lambda c: score(ss, *c))
            out[sh] = {'layer': best[0], 'tint': list(best[1]) if best[1] else None,
                       'n': len(ss), 'heldout_err': round(rule_err, 4), 'default_err': round(base_err, 4)}
            gain.append((len(ss), base_err, rule_err))
        print(f'{sh} n={len(ss):4d} default {base_err:.3f} rule {rule_err:.3f} {"KEEP" if sh in out else ""}')
    json.dump(out, open(os.path.join(WORK, 'layer_choice.json'), 'w'), indent=1)
    n = sum(g[0] for g in gain)
    if n:
        print(f'kept {len(out)} shader rules over {n} sections: held-out error '
              f'{sum(g[0]*g[1] for g in gain)/n:.3f} -> {sum(g[0]*g[2] for g in gain)/n:.3f}')


if __name__ == '__main__':
    main()
