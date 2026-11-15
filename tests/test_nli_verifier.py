"""
Tests for NLIVerifier dynamic model support and label mapping.
Verifies support for ygivenx/modernbert-base-mednli and any Hugging Face NLI model.
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import MedRagConfig
from medrag.retriever import Document, RetrievedDocument
from medrag.verifier import NLIVerifier, _HFSequenceClassificationWrapper


def test_nli_verifier_default_model_is_modernbert():
    verifier = NLIVerifier()
    assert verifier.model_name == "ygivenx/modernbert-base-mednli"
    cfg = MedRagConfig()
    assert cfg.nli_model == "ygivenx/modernbert-base-mednli"


def test_nli_verifier_custom_model_initialization():
    verifier = NLIVerifier(model_name="ygivenx/modernbert-base-mednli", batch_size=8)
    assert verifier.model_name == "ygivenx/modernbert-base-mednli"
    assert verifier.batch_size == 8
    assert verifier.idx_contradiction == 0
    assert verifier.idx_entailment == 1
    assert verifier.idx_neutral == 2


def test_nli_verifier_dynamic_label_mapping_standard_order():
    """Test resolution when model id2label is {0: contradiction, 1: entailment, 2: neutral}"""
    verifier = NLIVerifier(model_name="ygivenx/modernbert-base-mednli")
    mock_model = MagicMock()
    mock_model.model.config.id2label = {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }
    verifier._model = mock_model

    # Run label resolution logic as in _load
    cfg = mock_model.model.config
    id2label = cfg.id2label
    for idx, lbl in id2label.items():
        lbl_lower = str(lbl).lower()
        i = int(idx)
        if "contra" in lbl_lower:
            verifier.idx_contradiction = i
        elif "entail" in lbl_lower:
            verifier.idx_entailment = i
        elif "neut" in lbl_lower:
            verifier.idx_neutral = i

    assert verifier.idx_contradiction == 0
    assert verifier.idx_entailment == 1
    assert verifier.idx_neutral == 2


def test_nli_verifier_dynamic_label_mapping_mednli_order():
    """Test resolution when model id2label is {0: entailment, 1: neutral, 2: contradiction}"""
    verifier = NLIVerifier(model_name="ygivenx/modernbert-base-mednli")
    mock_model = MagicMock()
    mock_model.model.config.id2label = {
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
    }
    verifier._model = mock_model

    # Simulate _load resolution
    cfg = mock_model.model.config
    id2label = cfg.id2label
    for idx, lbl in id2label.items():
        lbl_lower = str(lbl).lower()
        i = int(idx)
        if "contra" in lbl_lower:
            verifier.idx_contradiction = i
        elif "entail" in lbl_lower:
            verifier.idx_entailment = i
        elif "neut" in lbl_lower:
            verifier.idx_neutral = i

    assert verifier.idx_entailment == 0
    assert verifier.idx_neutral == 1
    assert verifier.idx_contradiction == 2

    # Verify scoring with swapped indices
    mock_model.predict.return_value = np.array([[5.0, 1.0, 0.0]])  # High score on index 0 (entailment)
    scores = verifier._score_pairs([("premise", "hypothesis")])
    ent, con, neu = scores[0]
    assert ent > 0.90
    assert con < 0.10


def test_config_from_env_respects_nli_model():
    with patch.dict(os.environ, {"NLI_MODEL": "ygivenx/modernbert-base-mednli"}):
        cfg = MedRagConfig.from_env()
        assert cfg.nli_model == "ygivenx/modernbert-base-mednli"
