"""Typed experiment configuration: pydantic schemas, OmegaConf composition.

Each framework defines a :class:`Config` subclass — the schema mypy checks and
pydantic validates, so a mistyped knob fails before any GPU work. ``resolve``
layers a framework's named *variant* and command-line ``key=value`` overrides
onto its ``BASE`` mapping with OmegaConf (hydra's config engine: dotted paths,
type coercion), then validates the merged mapping into the schema. The
*validated* config is what :func:`~settlrl_learn.experiment.start_run` pins in
the run manifest, so reproduction starts from a checked object, not a loose dict.

This is the "pydra" seam (pydantic + OmegaConf) kept beside ``start_run`` rather
than under hydra's ``@hydra.main`` — the run directory and manifest are already
``bookkeeping``'s job, and hydra's working-directory takeover would fight it.
``model_dump()`` hands the heavier frameworks a plain dict so their internals
stay dict-threaded while the boundary stays typed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base for experiment config schemas: extra keys are an error (a typo in a
    variant or an override should fail loudly, not silently no-op)."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def resolve(
        cls,
        base: Mapping[str, Any],
        variant: Mapping[str, Any] | None = None,
        overrides: Sequence[str] = (),
    ) -> Self:
        """Merge ``base`` ◁ ``variant`` ◁ dotlist ``overrides`` (e.g.
        ``["games=4", "maximise.iterations=1"]``), then validate into ``cls``."""
        merged = OmegaConf.merge(
            OmegaConf.create(dict(base)),
            OmegaConf.create(dict(variant or {})),
            OmegaConf.from_dotlist(list(overrides)),
        )
        return cls.model_validate(OmegaConf.to_container(merged, resolve=True))

    def dump(self) -> dict[str, Any]:
        """JSON-able dict for the run manifest (and for dict-threaded bodies)."""
        return self.model_dump(mode="json")
