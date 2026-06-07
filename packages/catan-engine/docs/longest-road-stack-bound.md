# Bounding the longest-road DFS stack

`mechanics/longest_road.py` computes the longest road with an explicit-stack
DFS that pops frames in blocks of `_POP_K`. The stack lives in a fixed-size
array, so we need a static bound on how many live frames it can ever hold.
This note proves, with no prerequisites beyond the board's geometry:

> **peak ≤ S + min(S, K) + (M − 3)·K**

where M = 15 (`MAX_ROADS`, the most roads one player can own), S = 2M = 30
(the most seed frames), and K = `_POP_K`. For K = 32 that is **444**, and the
bound is *tight* in the abstract stack model (§4). `STACK_CAP = 445` adds one
scratch slot (`_DUMP`).

## 1. What the DFS does, abstractly

A *frame* is a partial trail: `d` distinct roads owned by the player, walked
end to end without reusing an edge, plus the vertex the walk currently stands
on (the *tip*). `d` is the frame's **depth**. One loop iteration:

1. **Pop** the top `m = min(K, sp)` frames.
2. **Expand** each popped frame: one child per legal one-road extension.
3. **Push** all children on top of the stack.

Four board facts constrain this — they are the *only* facts the proof uses:

- **(F1) At most 2 children per frame.** A vertex touches at most 3 edges,
  and the edge the trail arrived by is already used.
- **(F2) Depth caps at M = 15.** A trail's edges are distinct and owned. A
  frame at `d = 15` has no unused owned road left, hence no children.
- **(F3) Two children require `d ≤ 13`.** Pushing 2 children ("branching")
  needs two distinct *unused* owned edges at the tip, on top of the `d` used
  ones: `d + 2 ≤ M`. A frame at `d = 14` pushes at most one child
  ("chaining").
- **(F4) At most S = 2M = 30 seeds, all at depth 1** (at most one frame per
  direction per owned edge; the endpoint seeding keeps fewer).

Everything else about the board is abstracted into an *adversary*. A play of
the **stack game** starts from at most S items at depth 1; at every step the
adversary decides, per popped item at depth `d`, whether it emits 0, 1, or 2
children at depth `d + 1` (subject to F2/F3), and in what order the children
are pushed. Any run of the real DFS — any board, any occupancy, any player —
is one particular play, so a bound over all plays bounds the DFS. (Push order
within a block is adversarial too, so the code's cumsum-scatter layout is
covered for free.)

*Example (K = 2).* Stack `[A₂, B₂, C₃]` bottom-to-top (subscripts are
depths), sp = 3. The block pops `C₃, B₂`. Say C branches and B chains: push
`{c₄, c₄′, b₃}` in any order, e.g. `[A₂, b₃, c₄, c₄′]`, sp = 4. Note `A₂`
stayed *buried*: it can only be popped once everything above it is gone.

## 2. Three structural lemmas

**Lemma 1 (the floor).** Fix any moment and call the items present then the
*originals*. Ever after, the surviving originals sit contiguously at the
bottom of the stack, in their original order, below everything pushed since.

*Proof.* Items never move: pushes land strictly on top, pops take strictly
from the top. So nothing is ever inserted below a surviving original, and
originals are consumed top-down. ∎

Call a block that pops at least one original a **straddle**. A straddle
happens exactly when fewer than `m` non-originals sit above the floor.

**Lemma 2 (chains don't grow).** If every item popped during some stretch
emits at most 1 child, the stack size never increases during that stretch:
each block pops `m` and pushes at most `m`. ∎

**Lemma 3 (regions run standalone).** Between two consecutive straddles, the
part of the stack above the floor (the *region*) evolves exactly like an
independent play of the stack game started from the children the first
straddle pushed.

*Proof.* A non-straddle block touches only the region, so it pops
`m = min(K, sp)` items entirely from it, forcing `region ≥ m`. If
`K ≤ region` then `m = K`; otherwise `m = K` would make the block a straddle
unless `sp < K`, in which case `m = sp` and the floor must be empty, so
`m = region`. Either way `m = min(K, region)` — exactly the standalone pop. ∎

## 3. The theorem

Let `L(d) = max(0, 14 − d)` — the number of branchable depths in
`{d, …, 13}`.

**Theorem.** Any play starting from `n` items all at depth ≥ `d ≥ 2` peaks
at ≤ `n + K·L(d)`. The full game (S seeds at depth 1) peaks at
≤ `S + min(S, K) + K·L(2)` = `S + min(S, K) + 12K`.

**Proof** (downward induction on d).

*Base, d ≥ 14.* All items, and all their descendants, are at depth ≥ 14, so
by F2/F3 every pop emits at most one child. Lemma 2: peak = n. ✓

*Step, 2 ≤ d ≤ 13.* Assume the claim at d + 1. The initial n items are the
originals (Lemma 1); cut the play into **phases**, one per straddle.

*Phase 1.* Block 1 pops `p₁ = min(K, n)` originals and pushes `c₁ ≤ 2p₁`
children (F1), all at depth ≥ d + 1. By Lemma 3 the region then runs
standalone until the next straddle, so by induction its size stays
≤ `c₁ + K·L(d+1)`. Throughout phase 1:

    sp ≤ floor + region ≤ (n − p₁) + 2p₁ + K·L(d+1)
       = n + p₁ + K·L(d+1) ≤ n + K·(1 + L(d+1)) = n + K·L(d).

*Phase i ≥ 2* exists only when the floor is nonempty, i.e. `n > K` and
`p₁ = K`. Its straddle finds the old region worn down to a remnant
`r < m ≤ K`, pops all of it plus `kᵢ = m − r ≥ 1` originals, and pushes
`c ≤ 2m` children, again all at depth ≥ d + 1. As before, throughout
phase i:

    sp ≤ floorᵢ + c + K·L(d+1) ≤ (floorᵢ₋₁ − kᵢ) + 2(r + kᵢ) + K·L(d+1)
       = floorᵢ₋₁ + r + m + K·L(d+1)
       ≤ (n − K) + (K − 1) + K + K·L(d+1) = n + K·L(d) − 1,

using `floorᵢ₋₁ ≤ n − p₁ = n − K`. So every phase respects `n + K·L(d)`. ✓

*Top level.* The same phase arithmetic at d = 1, where F3 does not yet bind:
phase 1 gives `S + p₁ + K·L(2)` with `p₁ = min(S, K)`; phases i ≥ 2 give
`S + K + K·L(2) − 1` and only exist when S > K. Both are
≤ `S + min(S, K) + K·L(2)`. ∎

With M = 15, S = 30, K = 32: **peak ≤ 30 + 30 + 12·32 = 444**. (At K = 1,
the sequential DFS: 30 + 1 + 12 = 43.)

## 4. The bound is tight in the stack model

The adversary can reach 444 exactly — the *wave*, which branches a full block
at every level and buries the surplus:

| block | pops      | pushes | stack after (bottom → top)      | sp  |
|------:|-----------|--------|---------------------------------|----:|
| 1     | 30 @1     | 60 @2  | 60@2                            | 60  |
| 2     | 32 @2     | 64 @3  | 28@2, 64@3                      | 92  |
| 3     | 32 @3     | 64 @4  | 28@2, 32@3, 64@4                | 124 |
| ⋮     |           |        | net +32 per block               | ⋮   |
| 13    | 32 @13    | 64 @14 | 28@2, 32@3, …, 32@13, 64@14     | 444 |

Each block pops 32 of the top cohort's 64 frames, branches them all, and
buries the other 32 — net +32 for each of the 12 branchable levels 2…13.
The depth-14 cohort then only chains and dies. So no argument that sees only
F1–F4 can beat 444, and `STACK_CAP = 445` carries zero slack in the model.

The *real* DFS cannot play the wave — branching burns real edges, which is
why the fuzz test (`test_rules.py::test_dfs_peak_sp_stays_below_dump`)
observes peaks around 30. The fuzz test stays load-bearing regardless of this
proof: JAX silently drops out-of-bounds scatter updates, so an overflow would
corrupt results without raising.

## 5. Below 444: the resource model (open)

The gap between 444 and ~30 is about edge scarcity, which F1–F4 cannot see.
The sharper model gives each frame the resource **ρ = 2M − 2d − b**, where
`b` counts the *sibling edges* its lineage left behind when branching. Three
provable facts — a lineage branches at most once per vertex; walking into a
sibling edge dead-ends; an edge offers at most two sibling slots (one per
endpoint) — give `b ≤ 2(M − d)`, so ρ ≥ 0 for every frame that can still
branch. In ρ-terms a chain costs 2, a branch costs 3, and seeds start at
ρ₀ = 2M − 2 = 28.

Exhaustive small-instance searches (exact for K ≤ 3) and policy probes up to
K = 8 all match a growth rate of K per 3 resource, conjecturally

> peak ≤ S + min(S, K) + K·(⌊ρ₀/3⌋ − 1) ≈ **316**,

but this is unproven. The obstruction: chain fronts hold the top of the stack
at cost 2 per level while width is bought at cost 3 beneath them, so any
induction keyed on a region's *maximum* resource (like §3's) leaks back to
the cost-2 rate; and single-block potential arguments provably cannot work,
because transient overshoots are real (for ρ₀ ≡ 2 (mod 3) the exact peak
exceeds the law by up to K − 1). Mapping the real DFS onto the ρ-model also
needs care at the bottom: a frame whose two free edges are both spent
siblings pushes two children that die on arrival. Until someone closes this,
the practical cap stays 445.
