"""The Single-Observer ISMCTS tree (:func:`make_search` in
``search/__init__.py`` is its public wrapper).

:func:`make_tree` builds one search as a single XLA program over a
fixed-capacity node pool (the :class:`tree._Tree`), so it stays on device and
``vmap``s over lanes.

A *simulation* is one MCTS iteration over a freshly determinized world (so every
node's legal set is that world's true legality). Its phases:

  - determinize -- sample a world consistent with the belief;
  - select      -- descend the tree to an unexpanded edge (root by Sequential
                   Halving, interior by the improved-policy rule);
  - expand      -- attach the new leaf node;
  - evaluate    -- score the leaf with the value function (there is no rollout);
  - backup      -- add the leaf value to every edge on the path.

The result is the improved-policy weights ``softmax(root_logits + completed_Q)``
over the legal set; the caller supplies the root prior and takes the masked
argmax.

Module layout:

  - ``config``  -- :class:`SearchConfig`, the static ``_Cfg``, the
                   Sequential-Halving schedule, the tree dtype helpers;
  - ``tree``    -- the ``_Tree`` store and the select/expand/backup helpers;
  - ``descent`` -- the determinize/descend/evaluate walk and the engine seam;
  - ``loop``    -- ``_run`` and :func:`make_tree`.
"""

from __future__ import annotations

from ._types import TreeSearch
from .config import SearchConfig
from .loop import make_tree

__all__ = ["SearchConfig", "TreeSearch", "make_tree"]
