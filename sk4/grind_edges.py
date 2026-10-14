"""Find the edges you can grind, and hand them to the arena builder as collision splines.

Skate 3 does not decide what is grindable by looking at geometry. Grinds ride on SPLINES
carried in the collision arena, and the arena builder reads them from a `splines.json` beside
the GLBs, splitting them across tile boundaries itself. A converted map with no splines
has nothing to grind anywhere in it, however many rails and ledges it visibly contains --
which is why Riverside's rails did nothing.

So the rails have to be found. A grindable edge is a sharp convex crease with a surface on
top you could be riding when you reach it: the lip of a ledge, the top corner of a box, the
length of a handrail, a coping. That is a purely local test on a manifold edge --

  * shared by exactly two faces (an open boundary is a hole, not a rail),
  * CONVEX, so the material falls away either side rather than folding into a valley,
  * creased between `min_angle` and `max_angle` -- a right-angled ledge sits at 90 degrees,
    while a flat weld sits near 0 and a paper-thin fold near 180,
  * with at least one of its two faces facing up enough to stand on,
  * not so steep that it is a wall corner rather than a rail.

Edges that survive are chained into polylines, straightened, and returned in world space.
Every angle qualifies: nothing here cares which way the rail points.
"""

from __future__ import annotations

import numpy as np


def _weld(pos: np.ndarray, tol: float = 1e-3):
    """Merge vertices by position. Descenders meshes are unwelded face soups, so the
    index buffer alone reports no shared edges at all and every edge looks like a
    boundary."""
    q = np.round(np.asarray(pos, dtype=np.float64) / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return pos[first], inv.astype(np.int64)


def grind_edges(pos: np.ndarray, tris: np.ndarray, min_angle: float = 30.0,
                max_angle: float = 150.0, up_min: float = 0.45,
                max_edge_slope: float = 0.85, min_length: float = 1.2,
                max_splines: int = 400, straighten: float = 0.04):
    """Grindable polylines for one object, in the same space as `pos`. [] if none."""
    if len(tris) < 2 or len(pos) < 3:
        return []
    P, inv = _weld(pos)
    T = inv[np.asarray(tris)]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    # Drop duplicate faces. An instance that is drawn AND promoted to collision hands the
    # same triangles over twice; coincident copies give every edge four faces instead of
    # two, the manifold test rejects all of them, and an object full of ledges yields no
    # rails at all.
    if len(T):
        _, keep_first = np.unique(np.sort(T, axis=1), axis=0, return_index=True)
        T = T[np.sort(keep_first)]
    if len(T) < 2:
        return []

    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, c - a)
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-12
    if not ok.any():
        return []
    fn = np.where(ok[:, None], fn / np.where(ln[:, None] < 1e-12, 1.0, ln[:, None]), 0.0)

    # every triangle's three edges, keyed low->high so the two sides of an edge collide
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    opp = np.concatenate([T[:, 2], T[:, 0], T[:, 1]])          # the vertex not on the edge
    face = np.tile(np.arange(len(T)), 3)
    e = np.sort(e, axis=1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, opp, face = e[order], opp[order], face[order]
    same = (e[1:, 0] == e[:-1, 0]) & (e[1:, 1] == e[:-1, 1])
    pair = np.flatnonzero(same)
    if len(pair) == 0:
        return []
    # exactly two faces: a third would make it non-manifold, so skip runs longer than 2
    keep = np.ones(len(pair), dtype=bool)
    if len(pair) > 1:
        keep[1:] &= pair[1:] != pair[:-1] + 1
        keep[:-1] &= pair[:-1] + 1 != pair[1:]
    pair = pair[keep]
    if len(pair) == 0:
        return []

    i0, i1 = pair, pair + 1
    v0, v1 = e[i0, 0], e[i0, 1]
    n1, n2 = fn[face[i0]], fn[face[i1]]
    p2 = P[opp[i1]]

    ea, eb = P[v0], P[v1]
    d = eb - ea
    dlen = np.linalg.norm(d, axis=1)
    live = dlen > 1e-6
    if not live.any():
        return []

    # convex: the far vertex of the second face lies BEHIND the first face's plane
    convex = np.einsum("ij,ij->i", n1, p2 - ea) < -1e-4
    cosang = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    creased = (ang >= min_angle) & (ang <= max_angle)
    standable = (np.maximum(n1[:, 1], n2[:, 1]) >= up_min)
    slope = np.abs(d[:, 1]) / np.where(dlen < 1e-9, 1.0, dlen)
    rideable = slope <= max_edge_slope

    sel = live & convex & creased & standable & rideable
    if not sel.any():
        return []
    v0, v1 = v0[sel], v1[sel]

    # -- chain the surviving edges into polylines --------------------------
    adj: dict[int, list[int]] = {}
    for x, y in zip(v0.tolist(), v1.tolist()):
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    used: set[tuple[int, int]] = set()

    def walk(start: int, nxt: int) -> list[int]:
        chain = [start, nxt]
        used.add((min(start, nxt), max(start, nxt)))
        while True:
            here, prev = chain[-1], chain[-2]
            best, best_dot = None, -2.0
            dirv = P[here] - P[prev]
            nd = np.linalg.norm(dirv)
            if nd < 1e-9:
                break
            dirv = dirv / nd
            for cand in adj.get(here, ()):
                key = (min(here, cand), max(here, cand))
                if cand == prev or key in used:
                    continue
                w = P[cand] - P[here]
                nw = np.linalg.norm(w)
                if nw < 1e-9:
                    continue
                dot = float(np.dot(dirv, w / nw))
                if dot > best_dot:
                    best, best_dot = cand, dot
            if best is None or best_dot < 0.2:      # a corner sharper than ~78 deg ends it
                break
            used.add((min(here, best), max(here, best)))
            chain.append(best)
            if len(chain) > 4000:
                break
        return chain

    ends = [v for v, nb in adj.items() if len(nb) != 2]
    starts = ends if ends else list(adj.keys())
    out: list[np.ndarray] = []
    for s in starts:
        for nb in list(adj.get(s, ())):
            if (min(s, nb), max(s, nb)) in used:
                continue
            pts = P[walk(s, nb)]
            if len(pts) < 2:
                continue
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            if seg >= min_length:
                out.append(_straighten(pts, straighten))
            if len(out) >= max_splines:
                return out
    return out


def _straighten(pts: np.ndarray, tol: float) -> np.ndarray:
    """Drop points that sit on the line between their neighbours (Douglas-Peucker)."""
    if len(pts) <= 2 or tol <= 0:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = pts[j] - pts[i]
        L = np.linalg.norm(seg)
        if L < 1e-9:
            continue
        u = seg / L
        w = pts[i + 1:j] - pts[i]
        dist = np.linalg.norm(w - np.outer(w @ u, u), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return pts[keep]


def _smooth_line(pts: np.ndarray, limit: float, passes: int = 8) -> np.ndarray:
    """Pull the stair-steps out of a polyline without dropping its points.

    A rail traced round a rasterised deck steps by one cell at a time, so it zigzags at
    right angles even though the edge it follows is straight. Decimating it is the wrong
    cure -- keeping only the extreme points makes the corners SHARPER (measured: median turn
    18 deg -> 70 deg). Averaging each point with its neighbours instead pulls the steps into
    the line they approximate, and the clamp stops the rail drifting off the edge it belongs to.
    """
    if len(pts) < 3 or limit <= 0:
        return pts
    out = pts.astype(np.float64).copy()
    anchor = out.copy()
    for _ in range(passes):
        mid = (out[:-2] + out[2:]) * 0.5
        moved = out.copy()
        moved[1:-1] = (out[1:-1] + mid) * 0.5
        d = moved - anchor
        far = np.linalg.norm(d, axis=1)
        over = far > limit
        if over.any():
            d[over] *= (limit / far[over])[:, None]
        out = anchor + d
    return out.astype(np.float32)


def boundary_lines(pos: np.ndarray, tris: np.ndarray, min_length: float = 1.2,
                   max_edge_slope: float = 0.85, straighten: float = 0.04,
                   smooth: float = 0.0):
    """The open outline of a surface, as grindable polylines.

    A rebuilt deck stands a little PROUD of the boards it replaces, so the object's own
    top edges end up buried underneath it and the rail you could see is no longer the rail
    you can reach -- a smoothed ladder rides well and grinds nowhere. Its new ledge is the
    edge of the deck itself, which is an open boundary (one face, not two) and so invisible
    to the manifold test in `grind_edges`.
    """
    if len(tris) == 0 or len(pos) < 3:
        return []
    P, inv = _weld(pos)
    T = inv[np.asarray(tris)]
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e = e[order]
    same = np.r_[False, (e[1:, 0] == e[:-1, 0]) & (e[1:, 1] == e[:-1, 1])]
    # an edge is on the boundary when it appears exactly once
    keep = np.ones(len(e), dtype=bool)
    keep[same] = False
    keep[np.flatnonzero(same) - 1] = False
    b = e[keep]
    if len(b) == 0:
        return []
    d = P[b[:, 1]] - P[b[:, 0]]
    dl = np.linalg.norm(d, axis=1)
    live = dl > 1e-6
    slope = np.abs(d[:, 1]) / np.where(dl < 1e-9, 1.0, dl)
    b = b[live & (slope <= max_edge_slope)]
    if len(b) == 0:
        return []

    adj: dict[int, list[int]] = {}
    for x, y in zip(b[:, 0].tolist(), b[:, 1].tolist()):
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    used: set[tuple[int, int]] = set()
    out: list[np.ndarray] = []
    for s in list(adj):
        for nb in list(adj.get(s, ())):
            key = (min(s, nb), max(s, nb))
            if key in used:
                continue
            chain = [s, nb]
            used.add(key)
            while True:
                here, prev = chain[-1], chain[-2]
                nxt = None
                for c in adj.get(here, ()):
                    k = (min(here, c), max(here, c))
                    if c != prev and k not in used:
                        nxt = c
                        break
                if nxt is None:
                    break
                used.add((min(here, nxt), max(here, nxt)))
                chain.append(nxt)
                if len(chain) > 4000 or nxt == s:
                    break
            pts = P[chain]
            if len(pts) < 2:
                continue
            if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < min_length:
                continue
            # A closed ring has identical ends, and simplifying it as one run collapses it to
            # a zero-length pair. Cut it in half and keep both arcs -- each is a ledge.
            if len(pts) > 3 and float(np.linalg.norm(pts[0] - pts[-1])) < 1e-6:
                h = len(pts) // 2
                for part in (pts[:h + 1], pts[h:]):
                    if len(part) >= 2 and np.linalg.norm(np.diff(part, axis=0), axis=1).sum() >= min_length:
                        out.append(_straighten(_smooth_line(part, smooth), straighten))
            else:
                out.append(_straighten(_smooth_line(pts, smooth), straighten))
    return out
