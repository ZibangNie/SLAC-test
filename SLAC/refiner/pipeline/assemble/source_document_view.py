"""Immutable opt-in document rendering over saved dual-text atom traces."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from .source_atomizer import AtomizationTrace

Origin = Optional[tuple[int, int]]


class SourceDocumentViewError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _require(condition, code, message):
    if not condition:
        raise SourceDocumentViewError(code, message)


def _sequence(value, code):
    _require(isinstance(value, (list, tuple)), code, "expected a list or tuple")
    return tuple(value)


def _pair(value, code):
    pair = _sequence(value, code)
    _require(len(pair) == 2 and all(type(v) is int for v in pair), code, "expected two integer offsets")
    return pair


def _models(value):
    atoms = _sequence(value, "invalid_model_atoms")
    _require(all(isinstance(a, str) and a.strip() for a in atoms), "invalid_model_atoms", "model atoms must be nonblank strings")
    return atoms


def _origin_character_matches(char, source):
    # Validate the recorded scalar transform, without splitting or re-atomizing.
    if source == char:
        return True
    if char == "\n" and source in ("\r", "\r\n"):
        return True
    if char == " " and source and all(c in " \t\u00a0\u3000" for c in source):
        return True
    replacements = {"（": "(", "）": ")", "【": "[", "】": "]", "—": "-", "–": "-",
                    "－": "-", "“": '"', "”": '"', "‘": "'", "’": "'"}
    return len(source) == 1 and replacements.get(source) == char


@dataclass(frozen=True)
class DocumentSourceView:
    doc_id: str
    coordinate_system: str
    source_text: str
    model_atoms: tuple[str, ...]
    atom_char_spans: tuple[tuple[int, int], ...]
    model_char_origins: tuple[tuple[Origin, ...], ...]

    def __post_init__(self):
        _require(isinstance(self.doc_id, str) and self.doc_id.strip(), "invalid_doc_id", "document ID required")
        _require(isinstance(self.coordinate_system, str) and self.coordinate_system.strip(), "invalid_coordinate_system", "coordinate system required")
        _require(isinstance(self.source_text, str), "invalid_source", "source text must be a string")
        atoms = _models(self.model_atoms)
        _require(bool(atoms), "empty_document_atoms", "document rendering requires at least one atom")
        spans = tuple(_pair(p, "invalid_source_span") for p in _sequence(self.atom_char_spans, "invalid_source_span"))
        origins = tuple(tuple(None if origin is None else _pair(origin, "unsupported_origin")
                              for origin in _sequence(row, "unsupported_origin"))
                        for row in _sequence(self.model_char_origins, "unsupported_origin"))
        _require(len(atoms) == len(spans) == len(origins), "trace_model_mismatch", "atom metadata counts differ")
        cursor, previous_origin_end, covered = 0, 0, set()
        for atom, (start, end), row in zip(atoms, spans, origins, strict=True):
            _require(start == cursor and start < end <= len(self.source_text), "invalid_source_partition", "render spans must continuously partition source")
            _require(len(atom) == len(row), "unsupported_origin", "one origin required per model character")
            real_count = 0
            for char, origin in zip(atom, row, strict=True):
                if origin is None:
                    _require(char == " ", "unsupported_origin", "only inserted merge spaces may be originless")
                    continue
                left, right = origin
                _require(start <= left < right <= end, "unsupported_origin", "origin outside its render partition")
                _require(left >= previous_origin_end, "source_order_or_overlap", "origins reverse or overlap")
                _require(_origin_character_matches(char, self.source_text[left:right]), "unsupported_origin", "source interval does not support recorded scalar normalization")
                previous_origin_end = right
                covered.update(range(left, right))
                real_count += 1
            _require(real_count > 0, "unsupported_origin", "atom must contain source-backed characters")
            cursor = end
        _require(cursor == len(self.source_text), "invalid_source_partition", "render spans do not reach document end")
        _require(all(char.isspace() or i in covered for i, char in enumerate(self.source_text)),
                 "unrepresented_nonwhitespace", "source content lacks a model-character origin")
        object.__setattr__(self, "model_atoms", atoms)
        object.__setattr__(self, "atom_char_spans", spans)
        object.__setattr__(self, "model_char_origins", origins)

    @property
    def render_atoms(self) -> tuple[str, ...]:
        return tuple(self.source_text[start:end] for start, end in self.atom_char_spans)

    def char_span(self, start_atom: int, end_atom: int) -> tuple[int, int]:
        _require(type(start_atom) is int and type(end_atom) is int
                 and 0 <= start_atom < end_atom <= len(self.model_atoms),
                 "invalid_atom_span", "expected a nonempty in-range atom span")
        return self.atom_char_spans[start_atom][0], self.atom_char_spans[end_atom-1][1]

    def render(self, start_atom: int, end_atom: int) -> str:
        start, end = self.char_span(start_atom, end_atom)
        return self.source_text[start:end]

    def validate_against(self, doc_id: str, atoms: Sequence[str]) -> None:
        _require(doc_id == self.doc_id, "doc_id_mismatch", "document identity differs")
        _require(_models(atoms) == self.model_atoms, "model_atoms_mismatch", "cached model atom text or order differs")


def compose_document_source_view(*, doc_id: str, source_text: str, coordinate_system: str,
                                 unit_traces: Sequence[dict], model_atoms: Sequence[str]) -> DocumentSourceView:
    """Compose verified local traces; interunit whitespace belongs to the left atom.

    No atomization, substring alignment, fuzzy matching, or content lookup occurs.
    Local partitions and origins are validated before shifting to document offsets.
    """
    _require(isinstance(source_text, str), "invalid_source", "source text must be a string")
    expected_models = _models(model_atoms)
    entries = _sequence(unit_traces, "invalid_unit_traces")
    seen_ids, flat_models, starts, global_origins, previous_end = set(), [], [], [], 0
    for entry in entries:
        _require(isinstance(entry, dict) and {"native_unit_id", "source_span", "trace"} <= set(entry),
                 "invalid_unit_trace", "unit trace fields missing")
        uid = entry["native_unit_id"]
        _require(isinstance(uid, str) and uid.strip() and uid not in seen_ids, "invalid_unit_id", "unit IDs must be distinct nonblank strings")
        seen_ids.add(uid)
        start, end = _pair(entry["source_span"], "invalid_source_span")
        _require(0 <= previous_end <= start <= end <= len(source_text), "source_order_or_overlap", "unit source spans reverse, overlap, or exceed the document")
        gap = source_text[previous_end:start]
        _require(not gap or gap.isspace(), "nonwhitespace_gap", "unrepresented interunit source content")
        value = entry["trace"]
        trace = asdict(value) if isinstance(value, AtomizationTrace) else value
        _require(isinstance(trace, dict) and {"source_text", "model_atoms", "atoms", "operations", "trivia_only"} <= set(trace),
                 "invalid_trace", "expected AtomizationTrace or its serialized fields")
        local_source = source_text[start:end]
        _require(trace["source_text"] == local_source, "trace_source_mismatch", "trace source differs from the document slice")
        local_models = _models(trace["model_atoms"])
        atoms = _sequence(trace["atoms"], "invalid_trace")
        _require(type(trace["trivia_only"]) is bool and isinstance(trace["operations"], (list, tuple))
                 and all(isinstance(op, dict) for op in trace["operations"]), "invalid_trace", "invalid trace flags or operation records")
        if not local_models:
            _require(not atoms and trace["trivia_only"] and not local_source.strip(),
                     "invalid_trivia_trace", "only whitespace-only traces may omit atoms")
        else:
            _require(not trace["trivia_only"] and len(atoms) == len(local_models), "trace_model_mismatch", "trace atom count or trivia flag differs")
            spans, origins = [], []
            for atom, model in zip(atoms, local_models, strict=True):
                _require(isinstance(atom, dict) and {"model_text", "model_char_origins", "source_span", "source_text"} <= set(atom),
                         "invalid_trace", "atom trace fields missing")
                a, b = _pair(atom["source_span"], "invalid_source_span")
                _require(atom["model_text"] == model, "trace_model_mismatch", "trace model atom differs")
                _require(atom["source_text"] == local_source[a:b], "trace_source_mismatch", "atom source slice differs")
                spans.append((a, b))
                origins.append(atom["model_char_origins"])
            local = DocumentSourceView(doc_id, coordinate_system, local_source, local_models, spans, origins)
            flat_models.extend(local.model_atoms)
            starts.extend(start+a for a, _ in local.atom_char_spans)
            global_origins.extend(tuple(None if origin is None else (start+origin[0], start+origin[1])
                                        for origin in row) for row in local.model_char_origins)
        previous_end = end
    suffix = source_text[previous_end:]
    _require(not suffix or suffix.isspace(), "nonwhitespace_gap", "unrepresented trailing source content")
    _require(tuple(flat_models) == expected_models, "model_atoms_mismatch", "composed model atoms differ from cached atoms")
    _require(bool(starts), "empty_document_atoms", "entire document has no renderable atoms")
    cuts = [0] + starts[1:] + [len(source_text)]
    return DocumentSourceView(doc_id, coordinate_system, source_text, tuple(flat_models),
                              tuple(zip(cuts, cuts[1:])), tuple(global_origins))
