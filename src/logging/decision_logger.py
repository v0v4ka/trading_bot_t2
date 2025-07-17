"""
Agent Decision Logging System

This module provides structured logging for agent decisions in the trading bot.
It supports both standard and verbose logging modes, decision history tracking,
and log analysis utilities.
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import threading
from pathlib import Path


class LogLevel(Enum):
    """Log levels for decision events."""

    STANDARD = "standard"
    VERBOSE = "verbose"


class DecisionType(Enum):
    """Types of agent decisions."""

    SIGNAL_GENERATION = "signal_generation"
    TRADE_EXECUTION = "trade_execution"
    RISK_ASSESSMENT = "risk_assessment"
    DATA_ANALYSIS = "data_analysis"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    OTHER = "other"


@dataclass
class DecisionEvent:
    """
    Represents a single agent decision event.

    Attributes:
        agent_name: Name/identifier of the agent making the decision
        agent_type: Type/category of the agent
        timestamp: When the decision was made
        decision_type: Category of the decision
        action_taken: Description of the action/decision made
        confidence_score: Confidence level (0.0 to 1.0)
        reasoning_summary: Brief summary of the reasoning
        full_reasoning: Optional detailed reasoning (verbose mode)
        llm_prompt: Optional LLM prompt used (verbose mode)
        llm_response: Optional LLM response received (verbose mode)
        context_data: Additional context data as dictionary
    """

    agent_name: str
    agent_type: str
    timestamp: datetime
    decision_type: DecisionType
    action_taken: str
    confidence_score: float
    reasoning_summary: str
    full_reasoning: Optional[str] = None
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the decision event to a dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        data["timestamp"] = self.timestamp.isoformat()
        # Convert enum to string
        data["decision_type"] = self.decision_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionEvent":
        """Create a DecisionEvent from a dictionary."""
        # Convert string back to datetime
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        # Convert string back to enum
        data["decision_type"] = DecisionType(data["decision_type"])
        return cls(**data)


class DecisionLogger:
    """
    Logger for agent decisions with support for standard and verbose modes.

    This class provides structured logging of agent decisions with configurable
    verbosity levels and persistent storage.
    """

    def __init__(
        self,
        log_file_path: str = "logs/agent_decisions.log",
        log_level: LogLevel = LogLevel.STANDARD,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
    ):
        """
        Initialize the decision logger.

        Args:
            log_file_path: Path to the log file
            log_level: Default logging level (standard or verbose)
            max_file_size_mb: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.log_file_path = Path(log_file_path)
        self.log_level = log_level
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self._lock = threading.Lock()

        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        self._setup_logger()

    def _setup_logger(self):
        """Setup the file logger with rotation."""
        from logging.handlers import RotatingFileHandler

        self.logger = logging.getLogger(f"decision_logger_{id(self)}")
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create rotating file handler
        handler = RotatingFileHandler(
            self.log_file_path,
            maxBytes=self.max_file_size_mb * 1024 * 1024,
            backupCount=self.backup_count,
        )

        # JSON format for structured logging
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.propagate = False

    def log_decision(
        self,
        agent_name: str,
        agent_type: str,
        decision_type: DecisionType,
        action_taken: str,
        confidence_score: float,
        reasoning_summary: str,
        full_reasoning: Optional[str] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        log_level: Optional[LogLevel] = None,
    ) -> DecisionEvent:
        """
        Log a decision event.

        Args:
            agent_name: Name/identifier of the agent
            agent_type: Type/category of the agent
            decision_type: Category of the decision
            action_taken: Description of the action/decision made
            confidence_score: Confidence level (0.0 to 1.0)
            reasoning_summary: Brief summary of the reasoning
            full_reasoning: Optional detailed reasoning (verbose mode)
            llm_prompt: Optional LLM prompt used (verbose mode)
            llm_response: Optional LLM response received (verbose mode)
            context_data: Additional context data
            log_level: Override default log level for this entry

        Returns:
            The created DecisionEvent
        """
        # Validate confidence score
        if not 0.0 <= confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

        # Use provided log level or default
        effective_log_level = log_level or self.log_level

        # Create decision event
        event = DecisionEvent(
            agent_name=agent_name,
            agent_type=agent_type,
            timestamp=datetime.now(),
            decision_type=decision_type,
            action_taken=action_taken,
            confidence_score=confidence_score,
            reasoning_summary=reasoning_summary,
            full_reasoning=(
                full_reasoning if effective_log_level == LogLevel.VERBOSE else None
            ),
            llm_prompt=llm_prompt if effective_log_level == LogLevel.VERBOSE else None,
            llm_response=(
                llm_response if effective_log_level == LogLevel.VERBOSE else None
            ),
            context_data=context_data,
        )

        # Log to file
        with self._lock:
            log_entry = {
                "log_level": effective_log_level.value,
                "event": event.to_dict(),
            }
            self.logger.info(json.dumps(log_entry))

        return event

    def set_log_level(self, log_level: LogLevel):
        """Set the default log level."""
        self.log_level = log_level

    def get_log_file_path(self) -> Path:
        """Get the current log file path."""
        return self.log_file_path


class LogAnalyzer:
    """
    Utility class for analyzing agent decision logs.

    Provides methods to read, filter, and analyze decision logs.
    """

    def __init__(self, log_file_path: str = "logs/agent_decisions.log"):
        """
        Initialize the log analyzer.

        Args:
            log_file_path: Path to the log file to analyze
        """
        self.log_file_path = Path(log_file_path)

    def read_all_decisions(self) -> List[DecisionEvent]:
        """
        Read all decision events from the log file.

        Returns:
            List of DecisionEvent objects
        """
        decisions = []

        if not self.log_file_path.exists():
            return decisions

        try:
            with open(self.log_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            log_entry = json.loads(line)
                            if "event" in log_entry:
                                event = DecisionEvent.from_dict(log_entry["event"])
                                decisions.append(event)
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            # Skip malformed lines
                            continue
        except IOError:
            # File access error
            pass

        return decisions

    def filter_decisions(
        self,
        decisions: Optional[List[DecisionEvent]] = None,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        decision_type: Optional[DecisionType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
    ) -> List[DecisionEvent]:
        """
        Filter decision events based on criteria.

        Args:
            decisions: List of decisions to filter (if None, reads from file)
            agent_name: Filter by agent name
            agent_type: Filter by agent type
            decision_type: Filter by decision type
            start_time: Filter decisions after this time
            end_time: Filter decisions before this time
            min_confidence: Filter decisions with confidence >= this value
            max_confidence: Filter decisions with confidence <= this value

        Returns:
            Filtered list of DecisionEvent objects
        """
        if decisions is None:
            decisions = self.read_all_decisions()

        filtered = decisions

        if agent_name:
            filtered = [d for d in filtered if d.agent_name == agent_name]

        if agent_type:
            filtered = [d for d in filtered if d.agent_type == agent_type]

        if decision_type:
            filtered = [d for d in filtered if d.decision_type == decision_type]

        if start_time:
            filtered = [d for d in filtered if d.timestamp >= start_time]

        if end_time:
            filtered = [d for d in filtered if d.timestamp <= end_time]

        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence_score >= min_confidence]

        if max_confidence is not None:
            filtered = [d for d in filtered if d.confidence_score <= max_confidence]

        return filtered

    def get_decision_summary(
        self, decisions: Optional[List[DecisionEvent]] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of decision statistics.

        Args:
            decisions: List of decisions to analyze (if None, reads from file)

        Returns:
            Dictionary containing summary statistics
        """
        if decisions is None:
            decisions = self.read_all_decisions()

        if not decisions:
            return {
                "total_decisions": 0,
                "agents": {},
                "decision_types": {},
                "confidence_stats": {},
            }

        # Agent statistics
        agent_stats = {}
        for decision in decisions:
            agent_key = f"{decision.agent_name} ({decision.agent_type})"
            if agent_key not in agent_stats:
                agent_stats[agent_key] = 0
            agent_stats[agent_key] += 1

        # Decision type statistics
        decision_type_stats = {}
        for decision in decisions:
            dt = decision.decision_type.value
            if dt not in decision_type_stats:
                decision_type_stats[dt] = 0
            decision_type_stats[dt] += 1

        # Confidence statistics
        confidences = [d.confidence_score for d in decisions]
        confidence_stats = {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "count": len(confidences),
        }

        # Time range
        timestamps = [d.timestamp for d in decisions]
        time_range = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
        }

        return {
            "total_decisions": len(decisions),
            "time_range": time_range,
            "agents": agent_stats,
            "decision_types": decision_type_stats,
            "confidence_stats": confidence_stats,
        }

    def export_to_json(
        self,
        output_file: str,
        decisions: Optional[List[DecisionEvent]] = None,
        include_summary: bool = True,
    ) -> None:
        """
        Export decisions to a JSON file.

        Args:
            output_file: Path to output JSON file
            decisions: List of decisions to export (if None, reads from file)
            include_summary: Whether to include summary statistics
        """
        if decisions is None:
            decisions = self.read_all_decisions()

        export_data = {"decisions": [d.to_dict() for d in decisions]}

        if include_summary:
            export_data["summary"] = self.get_decision_summary(decisions)

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
