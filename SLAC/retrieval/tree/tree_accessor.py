from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from SLAC.retrieval.schemas.records import ChunkRecord, TreeAdjacencyRecord


@dataclass
class TreeAccessor:
    chunk_lookup: Dict[str, ChunkRecord]
    adjacency_lookup: Dict[str, TreeAdjacencyRecord]

    @classmethod
    def from_records(
        cls,
        chunks: List[ChunkRecord],
        adjacency_rows: List[TreeAdjacencyRecord],
    ) -> "TreeAccessor":
        return cls(
            chunk_lookup={c.chunk_id: c for c in chunks},
            adjacency_lookup={r.chunk_id: r for r in adjacency_rows},
        )

    def has_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self.chunk_lookup

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        return self.chunk_lookup.get(chunk_id)

    def get_adjacency(self, chunk_id: str) -> Optional[TreeAdjacencyRecord]:
        return self.adjacency_lookup.get(chunk_id)

    def get_parent_id(self, chunk_id: str) -> Optional[str]:
        row = self.get_adjacency(chunk_id)
        return row.parent_id if row else None

    def get_parent(self, chunk_id: str) -> Optional[ChunkRecord]:
        pid = self.get_parent_id(chunk_id)
        return self.get_chunk(pid) if pid else None

    def get_children_ids(self, chunk_id: str) -> List[str]:
        row = self.get_adjacency(chunk_id)
        return list(row.children_ids) if row else []

    def get_children(self, chunk_id: str) -> List[ChunkRecord]:
        return [self.chunk_lookup[cid] for cid in self.get_children_ids(chunk_id) if cid in self.chunk_lookup]

    def get_prev_id(self, chunk_id: str) -> Optional[str]:
        row = self.get_adjacency(chunk_id)
        return row.prev_chunk_id if row else None

    def get_next_id(self, chunk_id: str) -> Optional[str]:
        row = self.get_adjacency(chunk_id)
        return row.next_chunk_id if row else None

    def get_prev(self, chunk_id: str) -> Optional[ChunkRecord]:
        pid = self.get_prev_id(chunk_id)
        return self.get_chunk(pid) if pid else None

    def get_next(self, chunk_id: str) -> Optional[ChunkRecord]:
        nid = self.get_next_id(chunk_id)
        return self.get_chunk(nid) if nid else None

    def get_ancestor_ids(self, chunk_id: str, max_hops: int = 2) -> List[str]:
        out: List[str] = []
        cur = chunk_id
        for _ in range(max_hops):
            pid = self.get_parent_id(cur)
            if not pid:
                break
            out.append(pid)
            cur = pid
        return out

    def get_ancestors(self, chunk_id: str, max_hops: int = 2) -> List[ChunkRecord]:
        return [self.chunk_lookup[cid] for cid in self.get_ancestor_ids(chunk_id, max_hops=max_hops) if cid in self.chunk_lookup]

    def get_sibling_ids(
        self,
        chunk_id: str,
        left: int = 1,
        right: int = 1,
        include_self: bool = False,
    ) -> List[str]:
        row = self.get_adjacency(chunk_id)
        if row is None or not row.parent_id:
            return []

        siblings = self.get_children_ids(row.parent_id)
        if not siblings:
            return []

        try:
            idx = siblings.index(chunk_id)
        except ValueError:
            return []

        start = max(0, idx - max(left, 0))
        end = min(len(siblings), idx + max(right, 0) + 1)
        window = siblings[start:end]

        if not include_self:
            window = [x for x in window if x != chunk_id]
        return window

    def get_siblings(
        self,
        chunk_id: str,
        left: int = 1,
        right: int = 1,
        include_self: bool = False,
    ) -> List[ChunkRecord]:
        return [self.chunk_lookup[cid] for cid in self.get_sibling_ids(chunk_id, left=left, right=right, include_self=include_self) if cid in self.chunk_lookup]

    def get_neighbor_ids(
        self,
        chunk_id: str,
        left: int = 1,
        right: int = 1,
    ) -> List[str]:
        out: List[str] = []

        cur = chunk_id
        for _ in range(max(left, 0)):
            pid = self.get_prev_id(cur)
            if not pid:
                break
            out.append(pid)
            cur = pid

        right_ids: List[str] = []
        cur = chunk_id
        for _ in range(max(right, 0)):
            nid = self.get_next_id(cur)
            if not nid:
                break
            right_ids.append(nid)
            cur = nid

        return out + right_ids

    def get_neighbors(
        self,
        chunk_id: str,
        left: int = 1,
        right: int = 1,
    ) -> List[ChunkRecord]:
        return [self.chunk_lookup[cid] for cid in self.get_neighbor_ids(chunk_id, left=left, right=right) if cid in self.chunk_lookup]

    def get_subtree_ids(self, root_chunk_id: str, max_depth: int = 1) -> List[str]:
        """
        返回 root 的子树 chunk_id，不包含 root 自身。
        max_depth=1 -> children
        max_depth=2 -> children + grandchildren
        """
        if max_depth <= 0:
            return []

        out: List[str] = []
        frontier = [(root_chunk_id, 0)]
        visited: Set[str] = {root_chunk_id}

        while frontier:
            node_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue

            for child_id in self.get_children_ids(node_id):
                if child_id in visited:
                    continue
                visited.add(child_id)
                out.append(child_id)
                frontier.append((child_id, depth + 1))

        return out

    def get_local_branch_candidate_ids(
        self,
        seed_chunk_id: str,
        include_siblings: bool = True,
        include_children_of_parent: bool = True,
        max_subtree_depth: int = 1,
    ) -> List[str]:
        """
        用于 local branch reretrieve 的局部候选集：
        - parent 下的 siblings
        - 可选：seed parent 下其他 children
        - 不做全局 fan-out
        """
        out: List[str] = []

        parent_id = self.get_parent_id(seed_chunk_id)
        if not parent_id:
            return out

        if include_siblings:
            out.extend(self.get_sibling_ids(seed_chunk_id, left=9999, right=9999, include_self=False))

        if include_children_of_parent:
            for cid in self.get_children_ids(parent_id):
                if cid != seed_chunk_id and cid not in out:
                    out.append(cid)

        # 如果 parent 自己是一个标题节点，也允许 parent.children 的子层再看一层
        if max_subtree_depth > 1:
            extra: List[str] = []
            for cid in list(out):
                for gcid in self.get_subtree_ids(cid, max_depth=max_subtree_depth - 1):
                    if gcid != seed_chunk_id and gcid not in out and gcid not in extra:
                        extra.append(gcid)
            out.extend(extra)

        return out