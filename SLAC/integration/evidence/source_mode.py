"""Opt-in restoration of exact source passages before evidence selection.

This module does not select evidence, count tokens, render prompts, or invoke a
model. It preserves the existing normalizer's metadata/rank handling and uses
the retrieval source resolver to validate each supplied identity and text.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from SLAC.refiner.pipeline.assemble.source_coverage import NativeCoverageIndex
from SLAC.retrieval.dataio.source_records import resolve_refiner_source_selection
from SLAC.retrieval.schemas.records import ChunkRecord, RetrievalCandidate
from SLAC.integration.evidence.normalizers import normalize_candidate_to_selected_evidence
from SLAC.integration.io.schemas import SelectedEvidence


@dataclass(frozen=True)
class SourceModeConfig:
    chunk_lookup: Mapping[str, ChunkRecord]
    source_indexes: Mapping[str, NativeCoverageIndex]
    text_policy: Literal['exact', 'reconstruct']
    token_counter: Callable[[str], int]
    counter_version: str

    def __post_init__(self):
        if not isinstance(self.chunk_lookup, Mapping) or not isinstance(self.source_indexes, Mapping):
            raise TypeError('source mode requires chunk_lookup and source_indexes mappings')
        if not isinstance(self.text_policy, str) or self.text_policy not in ('exact', 'reconstruct'):
            raise ValueError('source mode text_policy must explicitly be exact or reconstruct')
        if not callable(self.token_counter):
            raise TypeError('source mode requires an explicit callable token_counter')
        if not isinstance(self.counter_version, str) or not self.counter_version.strip():
            raise ValueError('source mode counter_version must be a nonblank string')


def _identity(record, name):
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'source candidate requires a nonblank string {name}')
    return value


def _display_text(record):
    aliases = [record[name] for name in ('text', 'passage_text') if name in record]
    if not aliases or any(not isinstance(value, str) for value in aliases):
        raise ValueError('source candidate text/passage_text must be present string fields')
    if any(value != aliases[0] for value in aliases[1:]):
        raise ValueError('source candidate text/passage_text aliases must match exactly')
    return aliases[0]


def restore_source_candidates(
    records: Sequence[Mapping],
    *,
    source_mode: SourceModeConfig,
    query_id: str | None = None,
    query_text: str | None = None,
    source_name: str | None = None,
) -> list[SelectedEvidence]:
    """Restore each candidate independently, retaining list and rank semantics.

    Both text aliases, when supplied, must match before normalization. Exactly
    one lookup access per candidate is used; the existing resolver validates a
    one-entry mapping containing that same record. Duplicate candidates remain
    in the list for the existing selector's deduplication policy. The copied
    refiner_source metadata is the existing validated JSON snapshot, not a new
    provenance format. A fresh source lookup is required on every restoration.
    """
    if not isinstance(source_mode, SourceModeConfig):
        raise TypeError('source_mode must be a SourceModeConfig')
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise TypeError('source candidates must be a sequence of mappings')
    restored = []
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError('source candidate must be a mapping')
        chunk_id, doc_id = _identity(record, 'chunk_id'), _identity(record, 'doc_id')
        display_text = _display_text(record)
        try:
            lookup_record = source_mode.chunk_lookup[chunk_id]
        except KeyError as error:
            raise ValueError('source candidate is missing from chunk_lookup') from error
        candidate = RetrievalCandidate(chunk_id, doc_id, display_text, [], 0)
        _, snapshot, changed = resolve_refiner_source_selection(
            (), candidate, {chunk_id: lookup_record}, source_mode.source_indexes,
            text_policy=source_mode.text_policy)
        evidence = normalize_candidate_to_selected_evidence(
            dict(record), query_id=query_id, query_text=query_text,
            source_name=source_name, ordinal=ordinal, passage_text_override=snapshot.unit.text)
        # The legacy normalizer trims IDs; source identity must remain exact.
        evidence.chunk_id, evidence.doc_id = snapshot.unit.id, snapshot.unit.doc_id
        evidence.meta['refiner_source'] = deepcopy(lookup_record.meta['refiner_source'])
        evidence.meta['source_text_policy'] = source_mode.text_policy
        evidence.meta['source_text_changed'] = bool(changed)
        restored.append(evidence)
    return restored
