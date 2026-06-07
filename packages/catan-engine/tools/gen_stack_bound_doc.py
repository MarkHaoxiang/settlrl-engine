#!/usr/bin/env python3
"""Generate docs/longest-road-stack-bound.html (self-contained, inline SVG).

The page is committed; rerun this after editing and commit the result.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "longest-road-stack-bound.html"


def depth_color(d: int) -> str:
    # hue ramp blue (depth 2) -> orange (depth 14)
    hue = 215 - (d - 2) * 16
    return f"hsl({hue} 60% 52%)"


# ---------------------------------------------------------------- figure 1
def fig1() -> str:
    """One block iteration, K = 2."""
    BW, BH = 104, 30

    def box(x, y, label, d, dashed=False):
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        return (
            f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="5" '
            f'fill="{depth_color(d)}" stroke="var(--border)"{dash}/>'
            f'<text x="{x + BW / 2}" y="{y + BH / 2 + 5}" class="bx">{label}</text>'
        )

    s = ['<svg viewBox="0 0 660 252" role="img" aria-label="One block pop">']
    s.append(
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10 z" fill="var(--muted)"/></marker></defs>'
    )
    # before column
    bx = 44
    s.append(box(bx, 178, "A&#8322;", 2))
    s.append(box(bx, 144, "B&#8322;", 2))
    s.append(box(bx, 110, "C&#8323;", 3))
    s.append(f'<text x="{bx + BW / 2}" y="232" class="cap">before &#183; sp = 3</text>')
    # pop bracket around top two
    s.append(
        f'<rect x="{bx - 7}" y="104" width="{BW + 14}" height="76" rx="8" '
        'fill="none" stroke="var(--muted)" stroke-dasharray="5 4"/>'
    )
    s.append(f'<text x="{bx + BW / 2}" y="94" class="cap">pop m = min(K, sp) = 2</text>')
    # after column
    ax = 500
    s.append(box(ax, 178, "A&#8322;", 2))
    s.append(box(ax, 144, "b&#8323;", 3))
    s.append(box(ax, 110, "c&#8324;", 4))
    s.append(box(ax, 76, "c&#8324;&#8242;", 4))
    s.append(f'<text x="{ax + BW / 2}" y="232" class="cap">after &#183; sp = 4</text>')
    # arrows: C3 branches to c4, c4'; B2 chains to b3
    x0 = bx + BW + 10
    x1 = ax - 8
    s.append(
        f'<path d="M{x0} 122 C 300 110, 380 92, {x1} 91" fill="none" '
        'stroke="var(--muted)" stroke-width="1.6" marker-end="url(#arr)"/>'
    )
    s.append(
        f'<path d="M{x0} 128 C 300 128, 380 125, {x1} 125" fill="none" '
        'stroke="var(--muted)" stroke-width="1.6" marker-end="url(#arr)"/>'
    )
    s.append(
        f'<path d="M{x0} 161 C 300 161, 380 159, {x1} 159" fill="none" '
        'stroke="var(--muted)" stroke-width="1.6" marker-end="url(#arr)"/>'
    )
    s.append('<text x="330" y="84" class="cap">C&#8323; branches (two children, depth 4)</text>')
    s.append('<text x="330" y="178" class="cap">B&#8322; chains (one child, depth 3)</text>')
    s.append('<text x="330" y="218" class="cap small">A&#8322; stays buried until everything above it is popped</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- figure 2
def fig2() -> str:
    """Floor / straddle / phase schematic. n=10, K=4, L(d)=1, L(d+1)=0."""
    # state after block t: (floor, region)
    states = [
        (10, 0), (6, 8), (6, 7), (6, 5), (6, 3),
        (5, 8), (5, 6), (5, 4), (5, 2),
        (3, 8), (3, 6), (3, 3), (2, 8),
    ]
    T = len(states) - 1  # 12
    X0, X1 = 64, 626
    BASE, U = 212, 12.5

    def x(t: float) -> float:
        return X0 + (X1 - X0) * t / T

    def y(v: float) -> float:
        return BASE - U * v

    def step_path(tops: list[float], bottoms: list[float]) -> str:
        pts = []
        for t, v in enumerate(tops):
            pts.append(f"{x(t):.1f} {y(v):.1f}")
            if t < T:
                pts.append(f"{x(t + 1):.1f} {y(v):.1f}")
        for t in range(T, -1, -1):
            v = bottoms[t]
            if t < T:
                pts.append(f"{x(t + 1):.1f} {y(v):.1f}")
            pts.append(f"{x(t):.1f} {y(v):.1f}")
        return "M" + " L".join(pts) + " Z"

    floors = [f for f, _ in states]
    tops = [f + r for f, r in states]
    zeros = [0.0] * len(states)

    s = ['<svg viewBox="0 0 660 268" role="img" aria-label="Floor and phases">']
    # gridlines
    for v in (5, 10):
        s.append(
            f'<line x1="{X0}" y1="{y(v)}" x2="{X1}" y2="{y(v)}" '
            'stroke="var(--border)" stroke-width="1"/>'
        )
        s.append(f'<text x="{X0 - 8}" y="{y(v) + 4}" class="ax" style="text-anchor:end">{v}</text>')
    # areas
    s.append(f'<path d="{step_path(floors, zeros)}" fill="{depth_color(2)}" opacity="0.85"/>')
    s.append(f'<path d="{step_path(tops, floors)}" fill="{depth_color(7)}" opacity="0.55"/>')
    # bound line at 14 = n + K*L(d)
    s.append(
        f'<line x1="{X0}" y1="{y(14)}" x2="{X1}" y2="{y(14)}" '
        'stroke="var(--red)" stroke-width="1.6" stroke-dasharray="7 5"/>'
    )
    s.append(f'<text x="{X1}" y="{y(14) + 15}" class="ax" style="text-anchor:end" fill="var(--red)">n + K&#183;L(d) = 14</text>')
    # axis
    s.append(f'<line x1="{X0}" y1="{BASE}" x2="{X1}" y2="{BASE}" stroke="var(--muted)" stroke-width="1.2"/>')
    s.append(f'<text x="{(X0 + X1) / 2}" y="{BASE + 46}" class="ax">blocks &#8594;</text>')
    # straddle markers at t = 0,4,8,11 (the blocks producing states 1,5,9,12)
    for t in (0, 4, 8, 11):
        s.append(f'<text x="{x(t + 0.5)}" y="{BASE + 16}" class="ax">&#9650;</text>')
    s.append(f'<text x="{x(0.5)}" y="{BASE + 32}" class="ax">straddles pop into the floor</text>')
    # phase labels
    for label, t0, t1 in (("phase 1", 0.5, 4.5), ("phase 2", 4.5, 8.5), ("phase 3", 8.5, 11.5)):
        s.append(f'<text x="{(x(t0) + x(t1)) / 2}" y="34" class="ax">{label}</text>')
    for t in (4.5, 8.5, 11.5):
        s.append(
            f'<line x1="{x(t)}" y1="24" x2="{x(t)}" y2="{BASE}" '
            'stroke="var(--border)" stroke-width="1" stroke-dasharray="3 4"/>'
        )
    # area labels
    s.append(f'<text x="{x(2.2)}" y="{y(1.6)}" class="lab">floor (surviving originals)</text>')
    s.append(f'<text x="{x(2.2)}" y="{y(9.4)}" class="lab">region (all depth &#8805; d+1)</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- figure 3
def fig3() -> str:
    """The wave: stacked cohort columns reaching 444 (S=30, K=32, M=15)."""
    cols = [[(2, 60)]]
    for b in range(2, 14):
        segs = [(2, 28)] + [(d, 32) for d in range(3, b + 1)] + [(b + 1, 64)]
        cols.append(segs)
    X0, BASE = 70, 212
    CW, GAP = 30, 14
    SC = 200 / 445.0

    def y(v: float) -> float:
        return BASE - SC * v

    s = ['<svg viewBox="0 0 660 262" role="img" aria-label="The wave reaching 444">']
    for v in (0, 100, 200, 300, 400):
        s.append(
            f'<line x1="{X0 - 6}" y1="{y(v):.1f}" x2="636" y2="{y(v):.1f}" '
            'stroke="var(--border)" stroke-width="1"/>'
        )
        s.append(f'<text x="{X0 - 12}" y="{y(v) + 4:.1f}" class="ax" style="text-anchor:end">{v}</text>')
    # 444 line
    s.append(
        f'<line x1="{X0 - 6}" y1="{y(444):.1f}" x2="636" y2="{y(444):.1f}" '
        'stroke="var(--red)" stroke-width="1.6" stroke-dasharray="7 5"/>'
    )
    s.append(
        f'<text x="216" y="{y(444) + 16:.1f}" class="ax" style="text-anchor:start" '
        'fill="var(--red)">444 = S + min(S, K) + 12K</text>'
    )
    for i, segs in enumerate(cols):
        cx = X0 + 10 + i * (CW + GAP)
        acc = 0
        for d, cnt in segs:
            s.append(
                f'<rect x="{cx}" y="{y(acc + cnt):.1f}" width="{CW}" '
                f'height="{SC * cnt:.1f}" fill="{depth_color(d)}" '
                'stroke="var(--bg)" stroke-width="0.6"/>'
            )
            acc += cnt
        s.append(f'<text x="{cx + CW / 2}" y="{BASE + 16}" class="ax">{i + 1}</text>')
        if i == len(cols) - 1:
            s.append(f'<text x="{cx + CW / 2}" y="{y(acc) - 8:.1f}" class="ax" fill="var(--red)">peak</text>')
    s.append(f'<text x="{X0 + 10 + 6.5 * (CW + GAP)}" y="{BASE + 34}" class="ax">block &#8594;</text>')
    # depth legend: gradient ramp
    s.append(
        '<defs><linearGradient id="ramp" x1="0" y1="0" x2="1" y2="0">'
        + "".join(
            f'<stop offset="{(d - 2) / 12:.2f}" stop-color="{depth_color(d)}"/>'
            for d in range(2, 15)
        )
        + "</linearGradient></defs>"
    )
    s.append('<rect x="84" y="20" width="120" height="10" rx="3" fill="url(#ramp)"/>')
    s.append('<text x="84" y="44" class="ax" text-anchor="start">cohort depth 2 &#8594; 14</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- page
CSS = """
:root {
  --bg: #ffffff; --fg: #1c2025; --muted: #5d6673; --border: #d9dee6;
  --accent: #2563eb; --red: #c2403a; --panel: #f4f6fa; --eq: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #15181c; --fg: #e7eaee; --muted: #9aa4b0; --border: #2c343d;
    --accent: #6ea0ff; --red: #e06c66; --panel: #1c2127; --eq: #1a1f25;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 17px/1.65 Georgia, "Times New Roman", serif;
}
main { max-width: 46rem; margin: 0 auto; padding: 3rem 1.25rem 4rem; }
h1, h2 {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.25; letter-spacing: -0.01em;
}
h1 { font-size: 1.9rem; margin: 0 0 0.4rem; }
h2 { font-size: 1.25rem; margin: 2.6rem 0 0.8rem; }
.sub { color: var(--muted); margin: 0 0 1.8rem; }
.m { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.86em; }
.headline {
  background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--accent);
  border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1.4rem 0;
  font-size: 1.06em;
}
.box {
  background: var(--panel); border-left: 3px solid var(--accent);
  border-radius: 6px; padding: 0.7rem 1.1rem; margin: 1.1rem 0;
}
.box.red { border-left-color: var(--red); }
.box .tag {
  font-family: system-ui, sans-serif; font-weight: 600; font-size: 0.82em;
  text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent);
}
.box.red .tag { color: var(--red); }
.eq {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.84em; background: var(--eq); border-radius: 6px;
  padding: 0.7rem 1rem; margin: 0.9rem 0; overflow-x: auto; white-space: pre;
}
ul { padding-left: 1.3rem; }
li { margin: 0.45rem 0; }
figure { margin: 1.8rem 0; }
figure svg { width: 100%; height: auto; display: block; }
figcaption { color: var(--muted); font-size: 0.86em; text-align: center; margin-top: 0.5rem; }
svg text { font-family: system-ui, -apple-system, sans-serif; }
svg .bx { font-size: 14px; text-anchor: middle; fill: #fff; font-weight: 600; }
svg .cap { font-size: 12.5px; text-anchor: middle; fill: var(--muted); }
svg .cap.small { font-size: 11.5px; font-style: italic; }
svg .ax { font-size: 11.5px; text-anchor: middle; fill: var(--muted); }
svg .lab { font-size: 12.5px; text-anchor: middle; fill: #fff; font-weight: 600; }
.qed { float: right; color: var(--muted); }
footer {
  margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
  color: var(--muted); font-size: 0.84em;
}
a { color: var(--accent); }
"""

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bounding the longest-road DFS stack</title>
<style>__CSS__</style>
</head>
<body>
<main>

<h1>Bounding the longest-road DFS stack</h1>
<p class="sub">Why <span class="m">STACK_CAP = 445</span> in
<span class="m">mechanics/longest_road.py</span> &mdash; a self-contained proof.</p>

<p>The longest-road length is computed by an explicit-stack DFS that pops frames
in blocks of <span class="m">K = _POP_K</span>. The stack lives in a fixed-size
array, so we need a static bound on how many live frames it can ever hold. This
page proves, from the board&rsquo;s geometry alone:</p>

<div class="headline">
peak &le; S + min(S,&thinsp;K) + (M &minus; 3)&middot;K
<span style="color: var(--muted)">&emsp;where M = 15 roads, S = 2M = 30 seeds</span><br>
&emsp;&emsp;&thinsp;= <strong>444</strong> for K = 32 &mdash; and this is <em>tight</em> in the abstract stack model (&sect;4).
<span class="m">STACK_CAP = 445</span> adds one scratch slot (<span class="m">_DUMP</span>).
</div>

<h2>1&ensp;What the DFS does, abstractly</h2>

<p>A <em>frame</em> is a partial trail: <span class="m">d</span> distinct roads owned
by the player, walked end to end without reusing an edge, plus the vertex the walk
currently stands on (the <em>tip</em>). Call <span class="m">d</span> the frame&rsquo;s
<strong>depth</strong>. One loop iteration: <strong>pop</strong> the top
<span class="m">m = min(K, sp)</span> frames; <strong>expand</strong> each popped frame
into one child per legal one-road extension; <strong>push</strong> all children on top.</p>

<figure>__FIG1__
<figcaption>One block at K = 2. Subscripts are depths. The block pops the top two
frames; C&#8323; branches, B&#8322; chains, and the children land on top.</figcaption>
</figure>

<p>Four board facts constrain expansion &mdash; the <em>only</em> facts the proof uses:</p>
<ul>
<li><strong>F1 &mdash; at most 2 children per frame.</strong> A vertex touches at most
3 edges, and the edge the trail arrived by is already used.</li>
<li><strong>F2 &mdash; depth caps at M = 15.</strong> A trail&rsquo;s edges are distinct
and owned. A frame at d = 15 has no unused owned road left, hence no children.</li>
<li><strong>F3 &mdash; two children require d &le; 13.</strong> Branching needs two
distinct <em>unused</em> owned edges at the tip, on top of the d used ones:
d + 2 &le; M. A frame at d = 14 chains (&le; 1 child), never branches.</li>
<li><strong>F4 &mdash; at most S = 2M = 30 seeds, all at depth 1</strong> (at most one
frame per direction per owned edge; the endpoint seeding keeps fewer).</li>
</ul>

<p>Everything else about the board is abstracted into an <em>adversary</em>. A play of
the <strong>stack game</strong> starts from at most S items at depth 1; each step, the
adversary decides per popped item at depth d whether it emits 0, 1, or 2 children at
depth d + 1 (subject to F2/F3), and in what order children are pushed. Every run of
the real DFS &mdash; any board, any occupancy, any player &mdash; is one particular
play, so a bound over all plays bounds the DFS. Push order within a block is
adversarial too, so the code&rsquo;s cumsum-scatter layout is covered for free.</p>

<h2>2&ensp;Three structural lemmas</h2>

<div class="box"><span class="tag">Lemma 1 &middot; the floor</span><br>
Fix any moment and call the items present then the <em>originals</em>. Ever after, the
surviving originals sit contiguously at the bottom of the stack, in their original
order, below everything pushed since.<br>
<em>Proof.</em> Items never move: pushes land strictly on top, pops take strictly from
the top. So nothing is ever inserted below a surviving original, and originals are
consumed top-down. <span class="qed">&#8718;</span></div>

<p>Call a block that pops at least one original a <strong>straddle</strong>. A straddle
happens exactly when fewer than m non-originals sit above the floor.</p>

<div class="box"><span class="tag">Lemma 2 &middot; chains don&rsquo;t grow</span><br>
If every item popped during some stretch emits at most one child, the stack size never
increases during that stretch: each block pops m and pushes at most m.
<span class="qed">&#8718;</span></div>

<div class="box"><span class="tag">Lemma 3 &middot; regions run standalone</span><br>
Between two consecutive straddles, the part of the stack above the floor (the
<em>region</em>) evolves exactly like an independent play of the stack game started
from the children the first straddle pushed.<br>
<em>Proof.</em> A non-straddle block pops m = min(K, sp) items entirely from the
region, forcing region &ge; m. If K &le; region then m = K; otherwise m = K would make
the block a straddle unless sp &lt; K, in which case m = sp and the floor must be
empty, so m = region. Either way m = min(K, region) &mdash; exactly the standalone
pop. <span class="qed">&#8718;</span></div>

<h2>3&ensp;The theorem</h2>

<p>Let <span class="m">L(d) = max(0, 14 &minus; d)</span> &mdash; the number of
branchable depths in {d, &hellip;, 13}.</p>

<div class="box"><span class="tag">Theorem</span><br>
Any play starting from n items all at depth &ge; d &ge; 2 peaks at
&le; n + K&middot;L(d). The full game (S seeds at depth 1) peaks at
&le; S + min(S,&thinsp;K) + K&middot;L(2) = S + min(S,&thinsp;K) + 12K.</div>

<p><em>Proof &mdash; downward induction on d.</em></p>

<p><strong>Base, d &ge; 14.</strong> All items, and all their descendants, are at depth
&ge; 14, so by F2/F3 every pop emits at most one child. Lemma 2: peak = n.&ensp;&#10003;</p>

<p><strong>Step, 2 &le; d &le; 13.</strong> Assume the claim at d + 1. The initial n
items are the originals (Lemma 1); cut the play into <strong>phases</strong>, one per
straddle.</p>

<figure>__FIG2__
<figcaption>Schematic (n = 10, K = 4, L(d) = 1). Each straddle eats into the floor
and launches a fresh region; later phases start from a smaller floor, so their peaks
can&rsquo;t exceed phase&nbsp;1&rsquo;s headroom.</figcaption>
</figure>

<p><strong>Phase 1.</strong> Block 1 pops p&#8321; = min(K, n) originals and pushes
c&#8321; &le; 2p&#8321; children (F1), all at depth &ge; d + 1. By Lemma 3 the region
then runs standalone until the next straddle, so by induction its size stays &le;
c&#8321; + K&middot;L(d+1). Throughout phase 1:</p>

<div class="eq">sp &le; floor + region &le; (n &minus; p&#8321;) + 2p&#8321; + K&middot;L(d+1)
   = n + p&#8321; + K&middot;L(d+1) &le; n + K&middot;(1 + L(d+1)) = n + K&middot;L(d).</div>

<p><strong>Phase i &ge; 2</strong> exists only when the floor is nonempty, i.e.
n &gt; K and p&#8321; = K. Its straddle finds the old region worn down to a remnant
r &lt; m &le; K, pops all of it plus k&#7522; = m &minus; r &ge; 1 originals, and
pushes c &le; 2m children, again all at depth &ge; d + 1. As before, throughout
phase i:</p>

<div class="eq">sp &le; floor&#7522; + c + K&middot;L(d+1) &le; (floor&#7522;&#8331;&#8321; &minus; k&#7522;) + 2(r + k&#7522;) + K&middot;L(d+1)
   = floor&#7522;&#8331;&#8321; + r + m + K&middot;L(d+1)
   &le; (n &minus; K) + (K &minus; 1) + K + K&middot;L(d+1) = n + K&middot;L(d) &minus; 1,</div>

<p>using floor&#7522;&#8331;&#8321; &le; n &minus; p&#8321; = n &minus; K. So every phase
respects n + K&middot;L(d).&ensp;&#10003;</p>

<p><strong>Top level.</strong> The same phase arithmetic at d = 1, where F3 does not
yet bind: phase 1 gives S + p&#8321; + K&middot;L(2) with p&#8321; = min(S, K); phases
i &ge; 2 give S + K + K&middot;L(2) &minus; 1 and only exist when S &gt; K. Both are
&le; S + min(S,&thinsp;K) + K&middot;L(2). <span class="qed">&#8718;</span></p>

<p>With M = 15, S = 30, K = 32:&ensp;<strong>peak &le; 30 + 30 + 12&middot;32 =
444</strong>. (At K = 1, the old sequential DFS: 30 + 1 + 12 = 43.)</p>

<h2>4&ensp;The bound is tight in the stack model</h2>

<p>The adversary can reach 444 exactly &mdash; the <em>wave</em>, which branches a full
block at every level and buries the surplus. Each block pops 32 of the top
cohort&rsquo;s 64 frames, branches them all into 64 children one level deeper, and
buries the other 32 &mdash; net +32 for each of the 12 branchable levels 2&hellip;13.
The depth-14 cohort then only chains and dies.</p>

<figure>__FIG3__
<figcaption>The wave (S = 30, K = 32): stack composition after each block. Block 1
doubles the 30 seeds; blocks 2&ndash;13 each add a net +32 and bury half the previous
cohort; the colors are the buried cohorts by depth. Peak: 28 + 11&middot;32 + 64 = 444.</figcaption>
</figure>

<p>So no argument that sees only F1&ndash;F4 can beat 444, and
<span class="m">STACK_CAP = 445</span> carries zero slack in the model. The
<em>real</em> DFS cannot play the wave &mdash; branching burns real edges &mdash;
which is why the fuzz test
(<span class="m">test_rules.py::test_dfs_peak_sp_stays_below_dump</span>) observes
peaks around 30. The fuzz test stays load-bearing regardless of this proof: JAX
silently drops out-of-bounds scatter updates, so an overflow would corrupt results
without raising.</p>

<h2>5&ensp;Below 444: the resource model (open)</h2>

<div class="box red"><span class="tag">Conjecture</span><br>
peak &le; S + min(S,&thinsp;K) + K&middot;(&lfloor;&rho;&#8320;/3&rfloor; &minus; 1)
&asymp; <strong>316</strong>, where &rho;&#8320; = 2M &minus; 2 = 28.</div>

<p>The gap between 444 and the observed ~30 is about edge scarcity, which
F1&ndash;F4 cannot see. The sharper model gives each frame the resource
<span class="m">&rho; = 2M &minus; 2d &minus; b</span>, where b counts the
<em>sibling edges</em> its lineage left behind when branching. Three provable facts
&mdash; a lineage branches at most once per vertex; walking into a sibling edge
dead-ends; an edge offers at most two sibling slots (one per endpoint) &mdash; give
b &le; 2(M &minus; d), so &rho; &ge; 0 for every frame that can still branch. In
&rho;-terms a chain costs 2, a branch costs 3, and seeds start at &rho;&#8320; = 28.</p>

<p>Exhaustive small-instance searches (exact for K &le; 3) and policy probes up to
K = 8 all match a growth rate of K per 3 resource, but the law is unproven. The
obstruction: chain fronts hold the top of the stack at cost 2 per level while width
is bought at cost 3 beneath them, so any induction keyed on a region&rsquo;s
<em>maximum</em> resource (like &sect;3&rsquo;s) leaks back to the cost-2 rate; and
single-block potential arguments provably cannot work, because transient overshoots
are real (for &rho;&#8320; &equiv; 2 (mod 3) the exact peak exceeds the law by up to
K &minus; 1). Mapping the real DFS onto the &rho;-model also needs care at the
bottom: a frame whose two free edges are both spent siblings pushes two children
that die on arrival. Until someone closes this, the practical cap stays 445.</p>

<footer>
Source: <span class="m">src/catan_engine/mechanics/longest_road.py</span>
(<span class="m">STACK_CAP</span>) &middot; overflow guard:
<span class="m">tests/mechanics/test_rules.py::test_dfs_peak_sp_stays_below_dump</span>
&middot; June 2026. Self-contained page, no external assets.
</footer>

</main>
</body>
</html>
"""


def main() -> None:
    page = (
        HTML.replace("__CSS__", CSS)
        .replace("__FIG1__", fig1())
        .replace("__FIG2__", fig2())
        .replace("__FIG3__", fig3())
    )
    with open(OUT, "w") as f:
        f.write(page)
    print(f"wrote {OUT} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
