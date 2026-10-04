"""Opt-in dual-text atomization with provenance carried through each operation.

Model text must exactly match the production atomizer. Render partitions retain
the original source; they are not claimed to equal normalized model atoms.
Nothing in the default builder or pipeline calls this prototype.
The operation log includes evaluated candidates; final character origins and
render partitions are the authoritative correspondence for the returned atoms.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Optional

from . import build_refiner_input as builder

Origin = Optional[tuple[int, int]]


class SourceTraceError(ValueError):
    def __init__(self, code: str, stage: str, message: str):
        self.code, self.stage = code, stage
        super().__init__(f"{code} at {stage}: {message}")


@dataclass
class AtomizationTrace:
    source_text: str
    model_atoms: list[str]
    atoms: list[dict]
    operations: list[dict]
    trivia_only: bool


@dataclass(frozen=True)
class _Text:
    value: str
    origins: tuple[Origin, ...]

    def cut(self, start: int, end: int) -> "_Text":
        return _Text(self.value[start:end], self.origins[start:end])


class _Trace:
    def __init__(self, source: str, cfg: builder.RefinerInputBuildConfig):
        self.source, self.cfg = source, cfg
        self.operations: list[dict] = []

    def origin_union(self, origins: tuple[Origin, ...], stage: str) -> Origin:
        real = [origin for origin in origins if origin is not None]
        if not real:
            return None
        if len(real) != len(origins):
            raise SourceTraceError("unsupported_origin_composition", stage, "mixed synthetic and source characters")
        for left, right in zip(real, real[1:]):
            if right[0] < left[1]:
                raise SourceTraceError("source_overlap", stage, "normalization origins overlap or reverse")
            gap = self.source[left[1]:right[0]]
            if gap and not gap.isspace():
                raise SourceTraceError("unsupported_origin_composition", stage, "normalization would bridge source content")
        return real[0][0], real[-1][1]

    def replace(self, text: _Text, old: str, new: str, stage: str) -> _Text:
        # Exact matching here executes normalization, never post-hoc alignment.
        chars, origins, index = [], [], 0
        while index < len(text.value):
            if text.value.startswith(old, index):
                origin = self.origin_union(text.origins[index:index+len(old)], stage)
                chars.append(new)
                origins.append(origin)
                self.operations.append({"operation": "normalize_replace", "stage": stage,
                                        "from": old, "to": new, "source_span": origin})
                index += len(old)
            else:
                chars.append(text.value[index])
                origins.append(text.origins[index])
                index += 1
        return _Text("".join(chars), tuple(origins))

    def horizontal_spaces(self, text: _Text, stage: str) -> _Text:
        chars, origins, cursor = [], [], 0
        for match in re.finditer(r"[ \t]+", text.value):
            start, end = match.span()
            chars.append(text.value[cursor:start])
            origins.extend(text.origins[cursor:start])
            origin = self.origin_union(text.origins[start:end], stage)
            chars.append(" ")
            origins.append(origin)
            if text.value[start:end] != " ":
                self.operations.append({"operation": "collapse_horizontal_whitespace", "stage": stage,
                                        "source_span": origin, "input_characters": end-start})
            cursor = end
        chars.append(text.value[cursor:])
        origins.extend(text.origins[cursor:])
        return _Text("".join(chars), tuple(origins))

    def strip(self, text: _Text, stage: str) -> _Text:
        start, end = 0, len(text.value)
        while start < end and text.value[start].isspace():
            start += 1
        while end > start and text.value[end-1].isspace():
            end -= 1
        if start or end < len(text.value):
            self.operations.append({"operation": "strip_model_whitespace", "stage": stage,
                "removed_origins": list(text.origins[:start] + text.origins[end:])})
        return text.cut(start, end)

    def normalize(self, text: _Text, *, light: bool, stage: str) -> _Text:
        before = text.value
        replacements = [("\r\n", "\n"), ("\r", "\n")] if light else []
        replacements += [("\u00a0", " "), ("\u3000", " ")]
        if light:
            replacements += [("（", "("), ("）", ")"), ("【", "["), ("】", "]"),
                ("—", "-"), ("–", "-"), ("－", "-"), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]
        for old, new in replacements:
            text = self.replace(text, old, new, stage)
        text = self.strip(self.horizontal_spaces(text, stage), stage)
        expected = builder.normalize_text_light(before) if light else builder.normalize_spaces(before)
        if text.value != expected:
            raise SourceTraceError("builder_parity_mismatch", stage, "normalization differs")
        return text

    def lines(self, text: _Text) -> list[_Text]:
        parts, start = [], 0
        for end in [i for i, char in enumerate(text.value) if char == "\n"] + [len(text.value)]:
            part = self.normalize(text.cut(start, end), light=False, stage="sentence.line_candidate")
            if part.value:
                parts.append(part)
            start = end + 1
        return parts

    def sentences(self, text: _Text) -> list[_Text]:
        text = self.normalize(text, light=True, stage="sentence.normalize")
        if not text.value:
            return []
        lines = self.lines(text)
        if len(lines) > 1:
            headings = sum(builder._is_probably_heading_like(part.value) for part in lines)
            if headings >= max(1, len(lines)//2):
                self.operations.append({"operation": "sentence_line_fallback", "reason": "heading_like"})
                return lines
        ends = builder._safe_sentence_boundaries(text.value, dot_in_cjk=self.cfg.dot_in_cjk)
        if not ends:
            return lines if lines else [text]
        out, start = [], 0
        for end in ends:
            part = self.normalize(text.cut(start, end), light=False, stage="sentence.cut")
            if part.value:
                out.append(part)
            start = end
        tail = self.normalize(text.cut(start, len(text.value)), light=False, stage="sentence.tail")
        if tail.value:
            out.append(tail)
        if len(out) <= 1 and len(lines) > 1:
            return lines
        self.operations.append({"operation": "sentence_boundaries", "normalized_end_positions": ends})
        return out if out else [text]

    def too_long(self, text: str) -> bool:
        return len(text) > self.cfg.atom_max_chars or builder.approx_token_count(text) > self.cfg.atom_max_tokens

    def split_once(self, text: _Text) -> list[_Text]:
        text = self.normalize(text, light=False, stage="long.normalize")
        if not text.value:
            return []
        if not self.too_long(text.value):
            return [text]
        positions = builder._best_split_positions(text.value)
        cut = (min(positions, key=lambda pos: abs(pos-len(text.value)/2.0))
               if positions else len(text.value)//2)
        self.operations.append({"operation": "long_split", "normalized_cut": cut,
                                "traversal": "left_to_right_depth_first"})
        parts = [self.normalize(text.cut(a, b), light=False, stage="long.cut")
                 for a, b in ((0, cut), (cut, len(text.value)))]
        if not positions:
            return parts
        return [part for part in parts if part.value] or [text]

    def long_split(self, text: _Text) -> list[_Text]:
        pending = [self.normalize(text, light=False, stage="long.initial")]
        out = []
        while pending:
            current = pending.pop()
            if not current.value:
                continue
            if not self.too_long(current.value):
                out.append(current)
                continue
            parts = self.split_once(current)
            if len(parts) <= 1:
                out.append(current)
                continue
            pending.extend(reversed([part for part in parts if part.value]))
        return out

    def merged(self, left: _Text, right: _Text) -> tuple[_Text, bool]:
        synthetic_space = not builder.has_cjk(left.value + right.value)
        middle, origin = (" ", (None,)) if synthetic_space else ("", ())
        result = _Text(left.value + middle + right.value, left.origins + origin + right.origins)
        if synthetic_space:
            result = self.normalize(result, light=False, stage="merge.candidate")
        return result, synthetic_space

    def short_merge(self, atoms: list[_Text]) -> list[_Text]:
        out, index = [], 0
        while index < len(atoms):
            current = self.normalize(atoms[index], light=False, stage="merge.current")
            if not current.value:
                index += 1
                continue
            if not builder._is_short_atom(current.value, self.cfg):
                out.append(current)
                index += 1
                continue
            if index+1 < len(atoms):
                nxt = self.normalize(atoms[index+1], light=False, stage="merge.next")
                merged, inserted = self.merged(current, nxt)
                if not self.too_long(merged.value):
                    self.operations.append({"operation": "short_merge", "direction": "forward", "synthetic_space": inserted})
                    out.append(merged)
                    index += 2
                    continue
            if out:
                previous = out.pop()
                merged, inserted = self.merged(previous, current)
                if not self.too_long(merged.value):
                    self.operations.append({"operation": "short_merge", "direction": "backward", "synthetic_space": inserted})
                    out.append(merged)
                else:
                    out.extend((previous, current))
            else:
                out.append(current)
            index += 1
        return [atom for atom in out if atom.value.strip()]

    def atomize(self) -> list[_Text]:
        text = _Text(self.source, tuple((i, i+1) for i in range(len(self.source))))
        text = self.normalize(text, light=True, stage="initial")
        if not text.value:
            return []
        segments = self.sentences(text)
        if not segments and self.cfg.line_fallback:
            segments = self.lines(text)
        if not segments:
            segments = [text]
        atoms = [atom for segment in segments for atom in self.long_split(segment)]
        atoms = [self.normalize(atom, light=False, stage="before_merge") for atom in atoms]
        atoms = self.short_merge([atom for atom in atoms if atom.value])
        final = []
        for atom in atoms:
            atom = self.normalize(atom, light=False, stage="final")
            if atom.value:
                final.extend(self.long_split(atom) if self.too_long(atom.value) else [atom])
        return [atom for atom in final if atom.value.strip()]

    def finish(self, atoms: list[_Text]) -> AtomizationTrace:
        values = [atom.value for atom in atoms]
        if values != builder.atomize_unit_text(self.source, self.cfg):
            raise SourceTraceError("builder_parity_mismatch", "final", "model atom list differs")
        if not atoms:
            if self.source and not self.source.isspace():
                raise SourceTraceError("source_coverage_gap", "final", "nontrivia source produced no atoms")
            return AtomizationTrace(self.source, [], [], self.operations, True)
        covered, starts, previous_start, previous_end = set(), [], -1, 0
        for atom in atoms:
            if len(atom.value) != len(atom.origins):
                raise SourceTraceError("unsupported_origin_composition", "final", "model/origin lengths differ")
            real = []
            for char, origin in zip(atom.value, atom.origins):
                if origin is None:
                    if char != " ":
                        raise SourceTraceError("unsupported_origin_composition", "final", "synthetic character is not a merge space")
                    continue
                start, end = origin
                if not 0 <= start < end <= len(self.source):
                    raise SourceTraceError("unsupported_origin_composition", "final", "source interval out of bounds")
                if start < previous_start:
                    raise SourceTraceError("source_order_violation", "final", "origins reverse source order")
                if start < previous_end:
                    raise SourceTraceError("source_overlap", "final", "origins overlap")
                previous_start, previous_end = start, end
                covered.update(range(start, end))
                real.append(origin)
            if not real:
                raise SourceTraceError("unsupported_origin_composition", "final", "atom has no source characters")
            starts.append(real[0][0])
        if any(not char.isspace() and i not in covered for i, char in enumerate(self.source)):
            raise SourceTraceError("source_coverage_gap", "final", "nonwhitespace source character was lost")
        cuts = [0] + starts[1:] + [len(self.source)]
        result = []
        for atom, start, end in zip(atoms, cuts, cuts[1:]):
            if not start < end or any(origin is not None and not start <= origin[0] < origin[1] <= end for origin in atom.origins):
                raise SourceTraceError("source_overlap", "render", "render partition does not contain its model origins")
            result.append({"model_text": atom.value, "model_char_origins": list(atom.origins),
                           "source_span": (start, end), "source_text": self.source[start:end]})
        if "".join(atom["source_text"] for atom in result) != self.source:
            raise SourceTraceError("source_coverage_gap", "render", "render partitions do not reproduce source")
        self.operations.append({"operation": "render_partition", "separator_owner": "preceding_atom",
                                "outer_whitespace": "first_and_last_atom", "builder_parity": True})
        return AtomizationTrace(self.source, values, result, self.operations, False)


def trace_atomize_unit_text(source_text: str, cfg: Optional[builder.RefinerInputBuildConfig] = None) -> AtomizationTrace:
    """Trace one raw source unit without changing production model-facing atoms.

    Origins use Python character offsets. A many-to-one normalization character
    covers its complete source interval; None denotes an inserted merge space.
    Render spans form an exact source partition, not model-text character spans.
    Whitespace-only input retains source_text with trivia_only=True and no atoms;
    this does not claim admission to a source-mode projector with zero atoms.
    """
    if not isinstance(source_text, str):
        raise SourceTraceError("invalid_source", "input", "source_text must be a string")
    if cfg is not None and not isinstance(cfg, builder.RefinerInputBuildConfig):
        raise SourceTraceError("invalid_config", "input", "expected RefinerInputBuildConfig")
    cfg = deepcopy(cfg) if cfg is not None else builder.RefinerInputBuildConfig()
    for dimension in ("chars", "tokens"):
        maximum, minimum = getattr(cfg, "atom_max_"+dimension), getattr(cfg, "atom_min_"+dimension)
        if type(maximum) is not int or type(minimum) is not int or not 0 <= minimum <= maximum or maximum <= 0:
            raise SourceTraceError("invalid_config", "input", "expected positive maximum and 0 <= minimum <= maximum")
    for name in ("split_cjk_sentences", "split_en_sentences", "dot_in_cjk", "line_fallback",
                 "fix_empty_units", "prefer_merge_empty_to_right", "require_nonempty_atoms", "strict_validate"):
        if type(getattr(cfg, name)) is not bool:
            raise SourceTraceError("invalid_config", "input", "expected boolean configuration flags")
    trace = _Trace(source_text, cfg)
    return trace.finish(trace.atomize())
