"""
Structured logging and Agentic Reasoning Trace persistence for Muse-ic.

Tracks recommendation requests, semantic RAG retrieval events, weight
normalization adjustments, diversity penalties, guardrail interventions,
and multi-step agentic tool-call / self-correction traces.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "muse_ic.log"
AGENT_TRACE_FILE = LOG_DIR / "agent_reasoning_traces.log"


def get_logger(name: str = "muse_ic") -> logging.Logger:
    """Return a configured logger that writes structured entries to logs/muse_ic.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.addHandler(logging.NullHandler())

    return logger


def read_recent_logs(max_lines: int = 80) -> List[str]:
    """Read the most recent log lines from logs/muse_ic.log for UI observability."""
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines[-max_lines:]]
    except OSError:
        return []


def append_agent_trace(profile_label: str, query_summary: str, trace_steps: List[Dict[str, Any]]) -> None:
    """Persist a structured multi-step agent reasoning & tool-call trace to logs/agent_reasoning_traces.log."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 96,
            f"AGENTIC WORKFLOW TRACE | {timestamp} | Profile: {profile_label}",
            f"Query / Intent: {query_summary}",
            "-" * 96,
        ]
        for step in trace_steps:
            step_num = step.get("step", "?")
            phase = step.get("phase", "STEP")
            tool = step.get("tool", "none")
            thought = step.get("thought", "")
            observation = step.get("observation", "")
            decision = step.get("decision", "")
            lines.append(f"  [Step {step_num} | {phase}] Tool: `{tool}`")
            lines.append(f"    • Thought     : {thought}")
            lines.append(f"    • Observation : {observation}")
            if decision:
                lines.append(f"    • Decision    : {decision}")
        lines.append("=" * 96 + "\n")

        with open(AGENT_TRACE_FILE, mode="a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
