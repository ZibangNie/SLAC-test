"""Conservative, offline acronym links in the supplied native paragraph units.

The v1 rule accepts 2--6 ASCII capital letters in parentheses after the final N
ASCII words on that line, whose initials equal the acronym. Words may be joined
by whitespace or hyphens. No normalization or semantic inference is performed.
Uniqueness is scoped to supplied units, not certified for an unseen document.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import re

from .conditional import SourceRelation, Unit

RULE_VERSION = "native-explicit-acronym-v1"
_ACRONYM = re.compile(r"\(([A-Z]{2,6})\)")
_WORD = re.compile(r"[A-Za-z]+")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate(unit: Unit) -> None:
    if not isinstance(unit, Unit):
        raise TypeError("expected native Unit")
    if any(not isinstance(value, str) or not value.strip()
           for value in (unit.id, unit.doc_id, unit.text)):
        raise ValueError("unit identity and text must be nonempty strings")
    if type(unit.order) is not int or unit.order < 0:
        raise ValueError("native source order must be a nonnegative integer")


@dataclass(frozen=True)
class SourceAnchor:
    unit_id: str
    doc_id: str
    order: int
    text_sha256: str
    start: int
    end: int
    text: str

    def verify(self, native_units: Mapping[str, Unit]) -> Unit:
        unit = native_units[self.unit_id]
        _validate(unit)
        if type(self.order) is not int or self.order < 0:
            raise ValueError("anchor order must be a nonnegative integer")
        if (unit.id, unit.doc_id, unit.order, _hash(unit.text)) != (
                self.unit_id, self.doc_id, self.order, self.text_sha256):
            raise ValueError("native anchor identity, order or full text changed")
        if (type(self.start) is not int or type(self.end) is not int
                or not 0 <= self.start < self.end <= len(unit.text)
                or unit.text[self.start:self.end] != self.text):
            raise ValueError("native anchor slice changed")
        return unit


def _anchor(unit: Unit, start: int, end: int) -> SourceAnchor:
    return SourceAnchor(unit.id, unit.doc_id, unit.order, _hash(unit.text),
                        start, end, unit.text[start:end])


@dataclass(frozen=True)
class ExplicitDefinition:
    acronym: str
    anchor: SourceAnchor


@dataclass(frozen=True)
class DefinitionLink:
    definition: ExplicitDefinition
    mention: SourceAnchor

    def verify(self, native_units: Mapping[str, Unit]) -> SourceRelation:
        """Check both native endpoints before projecting core metadata.

        This checks local bindings and v1 syntax, not global uniqueness or truth.
        Uniqueness was determined over the extraction's supplied paragraph units.
        """
        source = self.definition.anchor.verify(native_units)
        target = self.mention.verify(native_units)
        if (source.id == target.id or source.doc_id != target.doc_id
                or source.order >= target.order):
            raise ValueError("definition must precede mention in the same document")
        if self.definition not in _definitions(source):
            raise ValueError("definition does not satisfy frozen extraction rule")
        match = _mention_pattern(self.definition.acronym).match(target.text, self.mention.start)
        if (self.mention.text != self.definition.acronym
                or match is None or match.span() != (self.mention.start, self.mention.end)
                or any(d.anchor.start <= self.mention.start < d.anchor.end
                       for d in _definitions(target))):
            raise ValueError("mention does not satisfy frozen extraction rule")
        return SourceRelation(target.id, source.id, "definition",
                              self.mention.start, self.mention.end)


@dataclass(frozen=True)
class DefinitionAmbiguity:
    doc_id: str
    acronym: str
    occurrences: tuple[ExplicitDefinition, ...]


@dataclass(frozen=True)
class DefinitionExtraction:
    definitions: tuple[ExplicitDefinition, ...]
    links: tuple[DefinitionLink, ...]
    ambiguities: tuple[DefinitionAmbiguity, ...]
    rule_version: str = RULE_VERSION


def _definitions(unit: Unit) -> tuple[ExplicitDefinition, ...]:
    found = []
    offset = 0
    for line in unit.text.splitlines(keepends=True):
        for match in _ACRONYM.finditer(line):
            acronym = match.group(1)
            prefix = line[:match.start()]
            words = list(_WORD.finditer(prefix))[-len(acronym):]
            if len(words) != len(acronym):
                continue
            if "".join(w.group()[0].upper() for w in words) != acronym:
                continue
            start = words[0].start()
            if start and (line[start - 1].isalnum() or line[start - 1] == "_"):
                continue  # Do not extract a suffix from a non-ASCII word.
            if any(not (c.isspace() or c == "-")
                   for a, b in zip(words, words[1:])
                   for c in line[a.end():b.start()]):
                continue
            if any(not c.isspace() for c in prefix[words[-1].end():]):
                continue
            found.append(ExplicitDefinition(acronym, _anchor(
                unit, offset + start, offset + match.end())))
        offset += len(line)
    return tuple(found)


def _mention_pattern(acronym: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(acronym)
                      + r"(?![A-Za-z0-9_])")


def extract_explicit_definitions(units: Sequence[Unit]) -> DefinitionExtraction:
    """Return immutable occurrences, later-unit links and ambiguous acronyms.

    Paragraph selection and document completeness belong to the caller. Input
    may be unordered; explicit source positions govern links. Duplicate IDs or
    document positions are rejected instead of silently discarded.
    """
    if not isinstance(units, Sequence):
        raise TypeError("units must be a sequence")
    ids, positions = set(), set()
    for unit in units:
        _validate(unit)
        position = (unit.doc_id, unit.order)
        if unit.id in ids or position in positions:
            raise ValueError("duplicate native unit identity or document order")
        ids.add(unit.id)
        positions.add(position)
    ordered = sorted(units, key=lambda u: (u.doc_id, u.order, u.id))
    definitions = tuple(d for unit in ordered for d in _definitions(unit))
    by_key, by_unit = defaultdict(list), defaultdict(list)
    for definition in definitions:
        by_key[(definition.anchor.doc_id, definition.acronym)].append(definition)
        by_unit[definition.anchor.unit_id].append(definition)
    links, ambiguities = [], []
    for (doc_id, acronym), occurrences in sorted(by_key.items()):
        if len(occurrences) != 1:
            ambiguities.append(DefinitionAmbiguity(doc_id, acronym, tuple(occurrences)))
            continue
        definition = occurrences[0]
        for unit in ordered:
            if unit.doc_id != doc_id or unit.order <= definition.anchor.order:
                continue
            for mention in _mention_pattern(acronym).finditer(unit.text):
                if any(d.anchor.start <= mention.start() < d.anchor.end
                       for d in by_unit[unit.id]):
                    continue
                links.append(DefinitionLink(definition, _anchor(unit, *mention.span())))
    return DefinitionExtraction(definitions, tuple(links), tuple(ambiguities))
