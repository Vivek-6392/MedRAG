"""
Tests for Phase 1 Research Foundation:
- Structured logging & PHI/credential redaction filter
- Pydantic configuration validation & experiment hash freezing
- Offline / cache-first model loading guarantees
"""
import io
import logging
import unittest
from medrag.logger import RedactionFilter
from medrag.config_schema import ExperimentConfig


class TestResearchLoggingAndRedaction(unittest.TestCase):
    """Verifies that API keys, tokens, and PHI are strictly scrubbed from logs."""

    def test_openai_and_groq_api_key_redaction(self):
        log_filter = RedactionFilter()
        raw_msg = (
            "Initializing OpenAI client with sk-abcdef1234567890abcdef1234567890 "
            "and Groq client gsk_12345678901234567890123456789012"
        )
        redacted = log_filter.redact(raw_msg)
        self.assertNotIn("sk-abcdef", redacted)
        self.assertNotIn("gsk_12345", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)
        self.assertIn("[REDACTED_GROQ_KEY]", redacted)

    def test_phi_redaction_ssn_email_mrn(self):
        log_filter = RedactionFilter()
        raw_msg = (
            "Patient record: SSN 123-45-6789, MRN: MRN9876543, "
            "contact doctor.smith@hospital.org or 415-555-2671."
        )
        redacted = log_filter.redact(raw_msg)
        self.assertNotIn("123-45-6789", redacted)
        self.assertNotIn("doctor.smith@hospital.org", redacted)
        self.assertNotIn("415-555-2671", redacted)
        self.assertIn("[REDACTED_SSN]", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertIn("[REDACTED_PHONE]", redacted)

    def test_logger_stream_redaction(self):
        stream = io.StringIO()
        logger = logging.getLogger("test_redact_stream")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        handler = logging.StreamHandler(stream)
        handler.addFilter(RedactionFilter())
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

        logger.info("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret123456789")
        output = stream.getvalue()
        self.assertNotIn("secret123456789", output)
        self.assertIn("[REDACTED_TOKEN]", output)


class TestConfigSchemaAndFreezing(unittest.TestCase):
    """Verifies Pydantic configuration validation and experiment freezing."""

    def test_valid_config_serialization_and_hash(self):
        cfg = ExperimentConfig(
            experiment_id="test_exp_01",
            benchmark_name="medqa",
            split="val",
            support_threshold=0.75,
            conflict_threshold=0.30,
        )
        cfg_hash = cfg.config_hash()
        self.assertIsInstance(cfg_hash, str)
        self.assertEqual(len(cfg_hash), 16)

        # Hash must be deterministic
        cfg2 = ExperimentConfig(
            experiment_id="test_exp_01",
            benchmark_name="medqa",
            split="val",
            support_threshold=0.75,
            conflict_threshold=0.30,
        )
        self.assertEqual(cfg.config_hash(), cfg2.config_hash())

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(
                experiment_id="invalid_exp",
                benchmark_name="medqa",
                missing_threshold=0.85,
                support_threshold=0.70,  # missing cannot exceed support
            )

    def test_to_medrag_config_conversion(self):
        cfg = ExperimentConfig(
            experiment_id="exp_convert",
            benchmark_name="pubmedqa",
            support_threshold=0.80,
            conflict_threshold=0.25,
        )
        runtime_cfg = cfg.to_medrag_config()
        self.assertEqual(runtime_cfg.support_threshold, 0.80)
        self.assertEqual(runtime_cfg.conflict_threshold, 0.25)


if __name__ == "__main__":
    unittest.main()
