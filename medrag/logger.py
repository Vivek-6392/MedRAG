"""
Structured Logging with Privacy Preserving Redaction Filter.
Ensures no API keys, credentials, or Protected Health Information (PHI)
are leaked into runtime or evaluation logs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional


class RedactionFilter(logging.Filter):
    """
    Sanitizes log records by redacting API tokens and potential PHI identifiers.
    """

    PATTERNS = [
        # API Keys & Secrets
        (re.compile(r"sk-[a-zA-Z0-9_-]{16,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
        (re.compile(r"gsk_[a-zA-Z0-9_-]{16,}", re.IGNORECASE), "[REDACTED_GROQ_KEY]"),
        (re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{16,}", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
        (re.compile(r"((?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?)[a-zA-Z0-9_\-\.]{8,}(['\"]?)", re.IGNORECASE), r"\1[REDACTED]\2"),
        
        # Protected Health Information (PHI)
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b(?:mrn|record\s*#?)\s*[:=]?\s*([a-zA-Z0-9]{6,12})\b", re.IGNORECASE), "MRN:[REDACTED_MRN]"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
    ]

    @classmethod
    def redact(cls, text: str) -> str:
        if not isinstance(text, str):
            return text
        sanitized = text
        for pattern, replacement in cls.PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.redact(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.redact(str(v)) for v in record.args)
        return True


class JsonFormatter(logging.Formatter):
    """Machine-readable JSON formatter for structured evaluation and pipeline logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            payload["data"] = record.extra_data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(
    name: str = "medrag",
    level: Optional[int] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Returns a configured logger with redaction filters attached.
    """
    logger = logging.getLogger(name)
    if level is None:
        env_level = os.environ.get("MEDRAG_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.addFilter(RedactionFilter())
        if json_format or os.environ.get("MEDRAG_LOG_JSON", "0") == "1":
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure parent loggers do not leak unredacted logs
    logger.propagate = False
    return logger
