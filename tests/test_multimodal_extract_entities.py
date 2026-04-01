"""Tests for LightRAG multimodal entity extraction and embedding payload routing."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.utils import EmbeddingFunc


def _make_multimodal_global_config() -> dict:
    return {
        "multimodal_entity_extract_func": AsyncMock(),
        "llm_model_max_async": 1,
        "addon_params": {},
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_multimodal_entities_parses_structured_output():
    from lightrag.operate import extract_multimodal_entities

    global_config = _make_multimodal_global_config()
    multimodal_func = global_config["multimodal_entity_extract_func"]
    multimodal_func.return_value = {
        "entities": [
            {
                "entity_name": "Graph Node",
                "entity_type": "Concept",
                "description": "A visually grounded node.",
            },
            {
                "entity_name": "Figure 1",
                "entity_type": "Artifact",
                "description": "The source figure.",
            },
        ],
        "relationships": [
            {
                "source_entity": "Graph Node",
                "target_entity": "Figure 1",
                "description": "Graph Node appears in Figure 1.",
                "keywords": ["visual-grounding", "diagram"],
                "weight": 0.8,
            }
        ],
    }

    chunk_results = await extract_multimodal_entities(
        chunks={
            "chunk-001": {
                "tokens": 8,
                "content": "Figure reference chunk",
                "full_doc_id": "doc-001",
                "chunk_order_index": 0,
                "file_path": "paper.pdf",
                "multimodal_payload": {
                    "item_type": "image",
                    "image": "/tmp/figure.png",
                    "text": "Figure 1 shows the graph node.",
                },
                "context_text": "Section 2 describes the graph.",
            }
        },
        global_config=global_config,
    )

    maybe_nodes, maybe_edges = chunk_results[0]
    assert set(maybe_nodes.keys()) == {"Graph Node", "Figure 1"}
    assert maybe_nodes["Graph Node"][0]["source_id"] == "chunk-001"
    assert maybe_nodes["Graph Node"][0]["file_path"] == "paper.pdf"
    assert ("Graph Node", "Figure 1") in maybe_edges
    assert maybe_edges[("Graph Node", "Figure 1")][0]["keywords"] == "visual-grounding, diagram"
    multimodal_func.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_multimodal_entities_accepts_json_string():
    from lightrag.operate import extract_multimodal_entities

    global_config = _make_multimodal_global_config()
    global_config["multimodal_entity_extract_func"].return_value = json.dumps(
        {
            "entities": [
                {
                    "entity_name": "Axis Label",
                    "entity_type": "Data",
                    "description": "A label in the chart.",
                }
            ],
            "relationships": [],
        }
    )

    chunk_results = await extract_multimodal_entities(
        chunks={
            "chunk-001": {
                "tokens": 4,
                "content": "Chart chunk",
                "full_doc_id": "doc-001",
                "chunk_order_index": 0,
                "multimodal_payload": {"item_type": "image", "image": "/tmp/chart.png"},
            }
        },
        global_config=global_config,
    )

    maybe_nodes, maybe_edges = chunk_results[0]
    assert "Axis Label" in maybe_nodes
    assert maybe_edges == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_nano_vdb_uses_embedding_content_override(tmp_path):
    seen_batches: list[list[object]] = []

    async def fake_embed(batch, **kwargs):
        seen_batches.append(list(batch))
        return np.array([[1.0, 0.0] for _ in batch], dtype=np.float32)

    storage = NanoVectorDBStorage(
        namespace="chunks",
        workspace="",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": -1.0,
            },
        },
        embedding_func=EmbeddingFunc(embedding_dim=2, func=fake_embed),
        meta_fields={"content"},
    )
    with (
        patch(
            "lightrag.kg.nano_vector_db_impl.get_update_flag",
            new=AsyncMock(return_value=type("Flag", (), {"value": False})()),
        ),
        patch(
            "lightrag.kg.nano_vector_db_impl.get_namespace_lock",
            return_value=asyncio.Lock(),
        ),
    ):
        await storage.initialize()
        await storage.upsert(
            {
                "chunk-001": {
                    "content": "Human readable chunk",
                    "embedding_content": {
                        "text": "machine payload",
                        "image": "/tmp/figure.png",
                    },
                }
            }
        )

    assert seen_batches == [[{"text": "machine payload", "image": "/tmp/figure.png"}]]
