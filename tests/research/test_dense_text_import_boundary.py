"""Verify text inspection stays offline while index builders retain their wiring."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_text_composers_import_without_model_or_index_backends():
    source = """
import importlib.abc, sys
blocked = {'torch', 'transformers', 'sentence_transformers', 'numpy', 'faiss'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise RuntimeError(fullname)
sys.meta_path.insert(0, Guard())
from SLAC.retrieval.index.build_leaf_dense import compose_leaf_retrieval_text
from SLAC.retrieval.index.build_chunk_dense import compose_chunk_retrieval_text
assert not blocked.intersection(sys.modules)
"""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-c', source], cwd=root,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('kind', ['leaf', 'chunk'])
def test_index_builder_keeps_encoder_and_save_wiring(kind, monkeypatch, tmp_path):
    from SLAC.retrieval.index.build_chunk_dense import build_chunk_dense_index, compose_chunk_retrieval_text
    from SLAC.retrieval.index.build_leaf_dense import build_leaf_dense_index, compose_leaf_retrieval_text
    from SLAC.retrieval.schemas.records import ChunkRecord, LeafRecord

    class Vectors:
        shape = (1, 3)
        ndim = 2

        def __len__(self):
            return 1

    vectors, index, calls = Vectors(), object(), []
    def encode(texts):
        calls.append(('encode', texts))
        return vectors

    def build(given):
        assert given is vectors
        calls.append(('build',))
        return index

    def save(given, path):
        assert given is index
        calls.append(('save_index', Path(path).name))

    def ids(given, path):
        calls.append(('save_ids', given, Path(path).name))

    monkeypatch.setitem(sys.modules, 'SLAC.retrieval.index.faiss_utils',
                        SimpleNamespace(build_flat_ip_index=build, save_faiss_index=save, save_id_map=ids))
    chunk = ChunkRecord('doc', 'chunk', 0, 0, 1, 'Body.', 1, ['Header'], 1,
                        path_text='Header', anchor_text='Owner anchor')
    leaf = LeafRecord('doc', 'leaf', 'chunk', 0, 0, 1, 'Body.', ['Header'], 1,
                      path_text='Header')
    if kind == 'leaf':
        expected_text, expected_id = compose_leaf_retrieval_text(leaf, chunk), 'leaf'
        result = build_leaf_dense_index([leaf], {'chunk': chunk}, SimpleNamespace(encode_texts=encode), tmp_path)
    else:
        expected_text, expected_id = compose_chunk_retrieval_text(chunk), 'chunk'
        result = build_chunk_dense_index([chunk], SimpleNamespace(encode_texts=encode), tmp_path)
    assert calls == [('encode', [expected_text]), ('build',), ('save_index', 'faiss.index'),
                     ('save_ids', [expected_id], 'id_map.npy')]
    assert result['num_items'] == 1 and result['embedding_dim'] == 3
