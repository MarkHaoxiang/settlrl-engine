"""Game logic: the single-game, traceable/vmappable rule helpers and action cores.

Each topical module holds both its sub-rule helpers and the action cores built
on them: ``dice`` (roll + production), ``placement`` (placement legality +
build), ``setup`` (snake order + setup placement), ``trade`` (maritime),
``development`` (dev-card plays incl. knight), ``robber`` (move robber +
discard + steal), and ``turn`` (end turn). ``awards`` holds Longest Road /
Largest Army. ``common`` is the shared vocabulary every core needs (result
codes, phase predicates, economy helpers), and ``action`` is the aggregation
layer on top: the unified ``(ActionType, ActionParams)`` dispatch, the flat
action table, and the switch-free legality enumerations.
"""
