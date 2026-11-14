"""
Regression tests for Part A: Centralized LLM Provider Resolver.
Covers:
- Auto-selection with GROQ_API_KEY, OPENAI_API_KEY, none (mock fallback).
- Explicit provider requests (groq, openai, ollama, mock).
- Provider-specific credentials, URLs, and model defaults.
- Missing credentials validation (clear ValueError).
- Verification that provider switches are recorded and never silent.
"""
import unittest

from medrag.provider import (
    ProviderResolver,
    ProviderType,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENAI_MODEL,
)


class TestProviderResolver(unittest.TestCase):

    def test_auto_selection_groq_when_key_present(self):
        env = {"GROQ_API_KEY": "gsk_test123456789012345678901234567890"}
        resolved = ProviderResolver.resolve(requested_provider="auto", env=env)
        self.assertEqual(resolved.backend, ProviderType.GROQ.value)
        self.assertEqual(resolved.api_key, "gsk_test123456789012345678901234567890")
        self.assertEqual(resolved.model, DEFAULT_GROQ_MODEL)
        self.assertTrue(resolved.is_auto_selected)
        self.assertIn("GROQ_API_KEY", resolved.selection_reason)

    def test_auto_selection_openai_when_key_present(self):
        env = {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}
        resolved = ProviderResolver.resolve(requested_provider="auto", env=env)
        self.assertEqual(resolved.backend, ProviderType.OPENAI.value)
        self.assertEqual(resolved.api_key, "sk-test123456789012345678901234567890")
        self.assertEqual(resolved.model, DEFAULT_OPENAI_MODEL)
        self.assertTrue(resolved.is_auto_selected)
        self.assertIn("OPENAI_API_KEY", resolved.selection_reason)

    def test_auto_selection_falls_back_to_mock_when_no_credentials(self):
        env = {}
        resolved = ProviderResolver.resolve(requested_provider="auto", env=env)
        self.assertEqual(resolved.backend, ProviderType.MOCK.value)
        self.assertTrue(resolved.is_auto_selected)
        self.assertIn("mock", resolved.selection_reason)

    def test_explicit_provider_selection_records_reason(self):
        env = {"OPENAI_API_KEY": "sk-key"}
        resolved = ProviderResolver.resolve(requested_provider="openai", env=env)
        self.assertEqual(resolved.backend, ProviderType.OPENAI.value)
        self.assertFalse(resolved.is_auto_selected)
        self.assertIn("Explicitly requested", resolved.selection_reason)

    def test_missing_credentials_raises_clear_error_for_groq(self):
        env = {}
        with self.assertRaises(ValueError) as ctx:
            ProviderResolver.resolve(requested_provider="groq", env=env)
        self.assertIn("GROQ_API_KEY is missing", str(ctx.exception))

    def test_missing_credentials_raises_clear_error_for_openai(self):
        env = {}
        with self.assertRaises(ValueError) as ctx:
            ProviderResolver.resolve(requested_provider="openai", env=env)
        self.assertIn("OPENAI_API_KEY is missing", str(ctx.exception))

    def test_unsupported_provider_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            ProviderResolver.resolve(requested_provider="unsupported_vendor")
        self.assertIn("Unsupported LLM provider", str(ctx.exception))

    def test_ollama_endpoints_and_model_resolution(self):
        env = {
            "OLLAMA_BASE_URL": "http://gpu-server:11434/v1",
            "OLLAMA_MODEL": "meditron:7b",
        }
        resolved = ProviderResolver.resolve(requested_provider="ollama", env=env)
        self.assertEqual(resolved.backend, "ollama")
        self.assertEqual(resolved.model, "meditron:7b")
        self.assertEqual(resolved.base_url, "http://gpu-server:11434/v1")


if __name__ == "__main__":
    unittest.main()
