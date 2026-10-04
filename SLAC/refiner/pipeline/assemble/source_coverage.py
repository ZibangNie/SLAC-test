"""Exact raw-coordinate coverage; these records make no semantic support claim."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from .source_document_view import DocumentSourceView


class SourceCoverageError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _require(condition, code, message):
    if not condition:
        raise SourceCoverageError(code, message)


def _span(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 2
             and all(type(x) is int for x in value),
             "invalid_source_span", "expected two integer character offsets")
    start, end = value
    _require(0 <= start < end, "invalid_source_span", "expected a nonempty source interval")
    return start, end


@dataclass(frozen=True)
class NativeCoverageHit:
    native_unit_id: str
    native_source_span: tuple[int, int]
    overlap_source_span: tuple[int, int]
    kind: str

    def __post_init__(self):
        _require(isinstance(self.native_unit_id, str) and self.native_unit_id.strip(),
                 "invalid_unit_id", "native unit ID required")
        native, overlap = _span(self.native_source_span), _span(self.overlap_source_span)
        _require(native[0] <= overlap[0] < overlap[1] <= native[1],
                 "invalid_overlap", "overlap must lie within the native unit")
        _require(self.kind == ("full" if native == overlap else "partial"),
                 "invalid_coverage_kind", "coverage kind must describe the exact raw interval")
        object.__setattr__(self, "native_source_span", native)
        object.__setattr__(self, "overlap_source_span", overlap)


@dataclass(frozen=True)
class SourceCoverage:
    source_char_span: tuple[int, int]
    text_sha256: str
    hits: tuple[NativeCoverageHit, ...]
    gap_chars: int

    def __post_init__(self):
        span = _span(self.source_char_span)
        _require(isinstance(self.hits, (list, tuple))
                 and all(isinstance(hit, NativeCoverageHit) for hit in self.hits),
                 "invalid_hits", "expected native coverage hits")
        hits = tuple(self.hits)
        _require(isinstance(self.text_sha256, str) and len(self.text_sha256) == 64
                 and all(c in "0123456789abcdef" for c in self.text_sha256),
                 "invalid_text_hash", "expected a lowercase SHA256 digest")
        cursor, covered, identities = span[0], 0, set()
        for hit in hits:
            left, right = hit.overlap_source_span
            _require(cursor <= left < right <= span[1] and hit.native_unit_id not in identities,
                     "invalid_hits", "hits must be distinct, ordered and within the covered span")
            cursor = right
            covered += right - left
            identities.add(hit.native_unit_id)
        _require(type(self.gap_chars) is int and self.gap_chars >= 0
                 and self.gap_chars == span[1] - span[0] - covered,
                 "invalid_gap_count", "gap count must exclude all native-unit characters")
        object.__setattr__(self, "source_char_span", span)
        object.__setattr__(self, "hits", hits)

    @property
    def full_native_unit_ids(self) -> tuple[str, ...]:
        return tuple(hit.native_unit_id for hit in self.hits if hit.kind == "full")

    @property
    def partial_native_unit_ids(self) -> tuple[str, ...]:
        return tuple(hit.native_unit_id for hit in self.hits if hit.kind == "partial")

    @property
    def exact_native_unit_id(self) -> str | None:
        if len(self.hits) == 1 and self.gap_chars == 0:
            hit = self.hits[0]
            if hit.kind == "full" and hit.native_source_span == self.source_char_span:
                return hit.native_unit_id
        return None


@dataclass(frozen=True)
class _NativeUnit:
    native_unit_id: str
    source_span: tuple[int, int]
    source_text: str


@dataclass(frozen=True)
class NativeCoverageIndex:
    view: DocumentSourceView
    native_units: tuple[_NativeUnit, ...]

    def __post_init__(self):
        _require(isinstance(self.view, DocumentSourceView), "invalid_view", "DocumentSourceView required")
        _require(isinstance(self.native_units, Sequence) and not isinstance(self.native_units, (str, bytes)),
                 "invalid_native_units", "native units must be a sequence")
        source, units, seen, cursor = self.view.source_text, [], set(), 0
        for entry in self.native_units:
            if isinstance(entry, _NativeUnit):
                uid, offsets, text = entry.native_unit_id, entry.source_span, entry.source_text
            else:
                _require(isinstance(entry, dict) and {"native_unit_id", "source_span", "source_text"} <= set(entry),
                         "invalid_native_unit", "native unit fields missing")
                uid, offsets, text = entry["native_unit_id"], entry["source_span"], entry["source_text"]
            _require(isinstance(uid, str) and uid.strip() and uid not in seen,
                     "invalid_unit_id", "native unit IDs must be distinct nonblank strings")
            _require(isinstance(text, str), "invalid_source_text", "native source text must be a string")
            _require(bool(text.strip()), "blank_native_unit", "blank units must be represented as whitespace gaps")
            start, end = _span(offsets)
            _require(cursor <= start < end <= len(source), "source_order_or_overlap",
                     "native intervals reverse, overlap or exceed the document")
            _require(source[start:end] == text, "source_text_mismatch", "native text differs from its exact source slice")
            _require(not source[cursor:start].strip(), "nonwhitespace_gap", "uncovered document text must be whitespace")
            units.append(_NativeUnit(uid, (start, end), text))
            seen.add(uid)
            cursor = end
        _require(not source[cursor:].strip(), "nonwhitespace_gap", "uncovered trailing text must be whitespace")
        object.__setattr__(self, "native_units", tuple(units))

    def cover_atoms(self, start_atom: int, end_atom: int) -> SourceCoverage:
        start, end = self.view.char_span(start_atom, end_atom)
        hits, cursor, gaps = [], start, 0
        for unit in self.native_units:
            left, right = max(start, unit.source_span[0]), min(end, unit.source_span[1])
            if left >= right:
                continue
            _require(not self.view.source_text[cursor:left].strip(), "nonwhitespace_gap", "uncovered rendered text must be whitespace")
            gaps += left - cursor
            overlap = (left, right)
            hits.append(NativeCoverageHit(unit.native_unit_id, unit.source_span, overlap,
                                          "full" if overlap == unit.source_span else "partial"))
            cursor = right
        _require(not self.view.source_text[cursor:end].strip(), "nonwhitespace_gap", "uncovered rendered text must be whitespace")
        gaps += end - cursor
        digest = sha256(self.view.render(start_atom, end_atom).encode("utf-8")).hexdigest()
        return SourceCoverage((start, end), digest, tuple(hits), gaps)


def build_native_coverage_index(view: DocumentSourceView, native_units: Sequence[dict]) -> NativeCoverageIndex:
    """Freeze exact native intervals. Blank native units are explicitly rejected.

    Whitespace outside native intervals remains gap coverage; whitespace inside a
    native interval belongs to that unit. No text matching or scoring occurs.
    """
    return NativeCoverageIndex(view, native_units)
