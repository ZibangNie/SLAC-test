from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Chunk0Unit:
    unit_id: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk0Doc:
    doc_id: str
    domain: str
    chunk0_units: List[Chunk0Unit]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "domain": self.domain,
            "chunk0_units": [u.to_dict() for u in self.chunk0_units],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk0Doc":
        if "doc_id" not in data:
            raise ValueError("Missing required field: doc_id")
        if "domain" not in data:
            raise ValueError("Missing required field: domain")
        if "chunk0_units" not in data:
            raise ValueError("Missing required field: chunk0_units")

        units_raw = data["chunk0_units"]
        if not isinstance(units_raw, list):
            raise ValueError("chunk0_units must be a list")

        units: List[Chunk0Unit] = []
        for i, item in enumerate(units_raw):
            if not isinstance(item, dict):
                raise ValueError(f"chunk0_units[{i}] must be a dict")
            if "unit_id" not in item or "text" not in item:
                raise ValueError(f"chunk0_units[{i}] must contain unit_id and text")
            units.append(
                Chunk0Unit(
                    unit_id=int(item["unit_id"]),
                    text=str(item["text"]),
                )
            )

        meta = data.get("meta", {})
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise ValueError("meta must be a dict if provided")

        return cls(
            doc_id=str(data["doc_id"]),
            domain=str(data["domain"]),
            chunk0_units=units,
            meta=meta,
        )


@dataclass
class UnitAtomSpan:
    unit_id: int
    start_atom: int
    end_atom: int  # half-open: [start_atom, end_atom)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AtomsB0Doc:
    doc_id: str
    domain: str
    chunk0_units: List[Chunk0Unit]
    atoms: List[str]
    unit2atom_span: List[UnitAtomSpan]
    b0: List[int]
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.atoms, list):
            raise ValueError("atoms must be a list")
        if not all(isinstance(x, str) for x in self.atoms):
            raise ValueError("all atoms must be strings")

        if len(self.atoms) == 0:
            if len(self.b0) != 0:
                raise ValueError("b0 must be empty when atoms is empty")
        else:
            if len(self.b0) != len(self.atoms) - 1:
                raise ValueError(
                    f"Invalid b0 length: expected {len(self.atoms) - 1}, got {len(self.b0)}"
                )

        prev_end = 0
        for i, span in enumerate(self.unit2atom_span):
            if span.start_atom < 0 or span.end_atom < span.start_atom:
                raise ValueError(f"Invalid span at index {i}: {span}")
            if i == 0:
                prev_end = span.start_atom
            if span.start_atom < prev_end:
                raise ValueError(f"Non-monotonic spans at index {i}: {span}")
            prev_end = span.end_atom

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "doc_id": self.doc_id,
            "domain": self.domain,
            "chunk0_units": [u.to_dict() for u in self.chunk0_units],
            "atoms": self.atoms,
            "unit2atom_span": [s.to_dict() for s in self.unit2atom_span],
            "b0": self.b0,
            "meta": self.meta,
        }