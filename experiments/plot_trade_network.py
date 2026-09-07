# -*- coding: utf-8 -*-
"""
Figure 1 (paper): raw trade-itinerary network, all 25 cities on a ring in
alphabetical order, edge width and label showing the symmetrized
itinerary count between each pair. Purely descriptive of the input data
-- no model, no prediction -- so it draws directly from src/data.py, the
same source every other script in this package uses.

The original version of this script (moved to legacy/Initial Graph.py)
hardcoded its own separate copy of the trade matrix and city list rather
than importing them. Verified identical, row for row, to
data.ANCIENT_MATRIX_DATA/ANCIENT_CITY_NAMES before switching this script
over -- so this produces the exact same figure as before, just sourced
correctly from the single copy of the data going forward, instead of a
second copy that could silently drift from it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import networkx as nx  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

from src import config, data  # noqa: E402


def _bezier_point(p0, p1, p2, t):
    """Point at parameter t (0..1) on the quadratic Bezier curve p0->p1->p2."""
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def main():
    names = data.ANCIENT_CITY_NAMES
    sym_matrix = data.ancient_symmetric_matrix()
    _, _, known_names, _ = data.ancient_known_lost_split()
    known_set = set(known_names)

    G = nx.Graph()
    for city in names:
        G.add_node(city, status='Known' if city in known_set else 'Unknown')
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            weight = sym_matrix[i, j]
            if weight > 0:
                G.add_edge(names[i], names[j], weight=weight)

    # Hub-and-spoke layout: Kanes is placed at the center rather than on the
    # ring. This is not just a decluttering trick -- Kanes genuinely is the
    # dominant hub of this network (highest total edge weight by a wide
    # margin, exactly as the paper's own text describes it), so putting it at
    # the center is a more honest representation of the network's actual
    # structure, and it also removes the single biggest source of long,
    # center-crossing chords: every one of Kanes's ~15+ edges becomes a short
    # spoke instead of a chord that used to cut across the whole ring.
    HUB = 'Kanes'
    ring_names = [n for n in names if n != HUB]
    ring_graph = nx.Graph()
    ring_graph.add_nodes_from(ring_names)
    pos = nx.circular_layout(ring_graph)
    pos[HUB] = np.array([0.0, 0.0])

    fig, ax = plt.subplots(figsize=(22, 22))

    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights)

    # City-name label positions, computed up front (moved earlier than the
    # original version of this script, which only computed these right
    # before drawing them) specifically so the edge-label collision check
    # below can treat them as exclusion zones too -- the first fix only
    # avoided the node MARKERS, which just pushed edge labels onto the
    # city-name TEXT instead.
    label_pos = {
        node: (coords[0] * 1.12, coords[1] * 1.12) if node != HUB else (0.0, -0.14)
        for node, coords in pos.items()
    }

    # (owner node, position, clearance radius) for everything an edge label
    # must avoid. Name labels get a wider clearance than node markers since
    # they're multi-character text (e.g. "Purushaddum"), not a small circle
    # -- a single point-radius check is an approximation, but a generous
    # radius compensates for that reasonably well in practice. The owner
    # name is kept (not just the bare position) so a long chord's own two
    # endpoints can be excluded when deciding whether ITS curve needs to
    # bulge out further to clear an unrelated node in between (see below).
    exclusion_zones = (
        [(node, np.array(p), 0.11) for node, p in pos.items()]
        + [(node, np.array(p), 0.17) for node, p in label_pos.items()]
    )
    LABEL_CLEARANCE = 0.045  # minimum distance between two edge labels

    def resolve_collisions(point, placed_labels):
        """Nudge `point` away from any node marker, city-name label, or
        already-placed edge label it's sitting too close to. Iterative
        rather than one-shot, because clearing one collision can create a
        new one against something else -- a handful of passes converges in
        practice given how sparse all these things are relative to the
        plot area."""
        point = np.array(point, dtype=float)
        for _ in range(10):
            moved = False
            for _, zone_p, clearance in exclusion_zones:
                d = point - zone_p
                dist = np.linalg.norm(d)
                if dist < clearance:
                    direction = d / dist if dist > 1e-6 else np.array([1.0, 0.0])
                    point = zone_p + direction * clearance
                    moved = True
            for other_p in placed_labels:
                d = point - other_p
                dist = np.linalg.norm(d)
                if dist < LABEL_CLEARANCE:
                    direction = d / dist if dist > 1e-6 else np.array([0.0, 1.0])
                    point = other_p + direction * LABEL_CLEARANCE
                    moved = True
            if not moved:
                break
        return point

    placed_labels = []

    # If collision-avoidance has to push a label further than this from its
    # true position on the edge, the connection back to that edge is no
    # longer visually obvious (this is exactly what was reported as "18 is
    # misplaced" / a label near Hahhum-Karahna sitting "randomly" -- busy
    # clusters of nodes force a big push, and a big push leaves the label
    # floating with nothing showing which edge it belongs to). A thin leader
    # line from the label back to its original anchor point removes that
    # ambiguity regardless of how far the push ends up being.
    LEADER_LINE_THRESHOLD = 0.05

    def draw_edge_label(point, weight):
        anchor = np.array(point, dtype=float)
        point = resolve_collisions(point, placed_labels)
        placed_labels.append(point)
        if np.linalg.norm(point - anchor) > LEADER_LINE_THRESHOLD:
            ax.plot([anchor[0], point[0]], [anchor[1], point[1]],
                     color='darkred', linewidth=0.6, alpha=0.6,
                     linestyle='-', zorder=9)
            ax.plot(*anchor, marker='o', markersize=2.5, color='darkred',
                     alpha=0.6, zorder=9)
        # zorder=10 and a fully OPAQUE white box: explicitly guarantees this
        # is the topmost thing at its position no matter what else (a node
        # circle, a city name, another edge's curve) ends up nearby -- the
        # earlier version left zorder unset on nodes/names, so there was no
        # guarantee a label wouldn't end up drawn UNDER something else even
        # when its center point itself was not colliding with anything.
        ax.text(point[0], point[1], str(int(weight)),
                 fontsize=9, color='darkred', fontweight='bold',
                 ha='center', va='center', zorder=10,
                 bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                           edgecolor='none', alpha=1.0))

    for (u, v) in G.edges():
        weight = G[u][v]['weight']
        lw = 0.5 + (weight / max_w) * 6
        p0, p2 = np.array(pos[u]), np.array(pos[v])

        if HUB in (u, v):
            # Spoke: straight line from the center to a ring node.
            ax.plot([p0[0], p2[0]], [p0[1], p2[1]], color='gray',
                     linewidth=lw, alpha=0.3, zorder=1)
            label_point = p0 + (p2 - p0) * 0.40
            draw_edge_label(label_point, weight)
        else:
            # Ring-to-ring edge (Kanes not involved): curved outward, same
            # technique as before, now with far fewer/shorter chords left to
            # clutter the middle since every Kanes edge is gone from this set.
            mid = (p0 + p2) / 2
            chord_len = np.linalg.norm(p2 - p0)
            norm = np.linalg.norm(mid)
            outward = mid / norm if norm > 1e-6 else np.array([1.0, 0.0])
            bulge = 0.32 * chord_len + 0.08

            # A chord that skips several ring nodes (e.g. Hahhum-Karahna,
            # which skips Hurama/Hattus/Hanaknak) has its midpoint sitting
            # on the same radial line as whichever ring node is roughly
            # halfway between its two endpoints -- by construction, not
            # coincidence. A fixed bulge that's generous enough for a
            # short, adjacent-node chord is nowhere near enough to clear
            # that in-between node for a long one, so the curve (and its
            # label) ends up passing right through an unrelated city's
            # marker. Grow the bulge specifically for THIS edge until its
            # midpoint clears every node/name it doesn't itself connect to.
            other_zones = [(zp, cl) for owner, zp, cl in exclusion_zones
                           if owner not in (u, v)]
            for _ in range(20):
                control = mid + outward * bulge
                label_point = _bezier_point(p0, control, p2, 0.5)
                blocked = any(np.linalg.norm(label_point - zp) < cl
                              for zp, cl in other_zones)
                if not blocked:
                    break
                bulge += 0.08

            control = mid + outward * bulge
            path = Path([tuple(p0), tuple(control), tuple(p2)],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            ax.add_patch(PathPatch(path, facecolor='none', edgecolor='gray',
                                    linewidth=lw, alpha=0.25, zorder=1))
            label_point = _bezier_point(p0, control, p2, 0.5)
            draw_edge_label(label_point, weight)

    known_nodes = [n for n, d in G.nodes(data=True) if d['status'] == 'Known']
    unknown_nodes = [n for n, d in G.nodes(data=True) if d['status'] == 'Unknown']
    # Explicit zorder on every layer from here on, so the stacking order is
    # guaranteed rather than left to matplotlib's per-artist-type defaults
    # (which is what let a handful of edge-count labels end up rendered
    # UNDER a node circle or city name in the previous version, even though
    # their center points weren't technically colliding with anything).
    nx.draw_networkx_nodes(G, pos, nodelist=known_nodes, node_color='#2E86C1',
                            node_size=1500, alpha=1, ax=ax).set_zorder(2)
    nx.draw_networkx_nodes(G, pos, nodelist=unknown_nodes, node_color='#E67E22',
                            node_size=1500, alpha=1, ax=ax).set_zorder(2)
    # Kanes drawn larger, on top, to visually read as the hub it is.
    nx.draw_networkx_nodes(G, pos, nodelist=[HUB], node_color='#2E86C1',
                            node_size=2600, alpha=1, ax=ax,
                            edgecolors='black', linewidths=1.5).set_zorder(2)

    name_labels = nx.draw_networkx_labels(G, label_pos, font_size=12,
                                           font_weight='bold', ax=ax)
    for text_obj in name_labels.values():
        text_obj.set_zorder(3)  # above node circles, below edge-count labels

    ax.set_title("Trade Network with Itinerary Counts", fontsize=20, pad=30)
    ax.axis('off')
    ax.set_aspect('equal')
    margin = 1.35
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(config.FIGURES_DIR, "Figure1 - Trade Network with Itinerary Counts.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
