# Agent Decision Logging System

This package provides comprehensive decision logging capabilities for the AI Multi-Agent Trading Bot's agent orchestration system.

## Overview

The Agent Decision Logging System enables structured logging of agent decisions with support for:
- **Standard and verbose logging modes**
- **Decision history tracking with timestamps**
- **Log analysis and filtering capabilities**
- **Export functionality for decision review**
- **Thread-safe operation for multi-agent environments**

## Quick Start

```python
from src.logging import DecisionLogger, DecisionType, LogLevel

# Create a decision logger
logger = DecisionLogger(
    log_file_path="logs/agent_decisions.log",
    log_level=LogLevel.STANDARD
)

# Log a decision
event = logger.log_decision(
    agent_name="SignalAgent_EURUSD",
    agent_type="TechnicalAnalysisAgent",
    decision_type=DecisionType.SIGNAL_GENERATION,
    action_taken="Generated BUY signal for EURUSD",
    confidence_score=0.85,
    reasoning_summary="Strong bullish indicators detected",
    context_data={"symbol": "EURUSD", "price": 1.0895}
)
```

## Key Components

### DecisionLogger
Main logging class that handles structured decision logging with configurable verbosity.

**Features:**
- JSON-based structured logging
- Standard vs verbose modes
- Automatic file rotation
- Thread-safe operation
- Configurable log file size and backup retention

### DecisionEvent
Immutable dataclass representing a single agent decision with full context.

**Attributes:**
- `agent_name`: Unique identifier for the agent
- `agent_type`: Category/type of the agent
- `timestamp`: When the decision was made
- `decision_type`: Category of decision (signal, trade, risk, etc.)
- `action_taken`: Description of the action/decision
- `confidence_score`: Confidence level (0.0 to 1.0)
- `reasoning_summary`: Brief reasoning summary
- `full_reasoning`: Detailed reasoning (verbose mode)
- `llm_prompt`: LLM prompt used (verbose mode)
- `llm_response`: LLM response received (verbose mode)
- `context_data`: Additional context as dictionary

### LogAnalyzer
Utility class for reading, filtering, and analyzing historical decision logs.

**Capabilities:**
- Read all decisions from log files
- Filter by agent, type, time range, confidence
- Generate summary statistics
- Export to JSON format
- Handle malformed log entries gracefully

### Decision Types
Predefined categories for agent decisions:
- `SIGNAL_GENERATION`: Trading signal decisions
- `TRADE_EXECUTION`: Trade execution decisions
- `RISK_ASSESSMENT`: Risk management decisions
- `DATA_ANALYSIS`: Data analysis decisions
- `STRATEGY_ADJUSTMENT`: Strategy modification decisions
- `OTHER`: Other decision types

## Logging Modes

### Standard Mode
Logs essential decision data without verbose details:
- Agent identification and timestamps
- Decision type and action taken
- Confidence score and reasoning summary
- Context data
- Excludes: full reasoning, LLM prompts/responses

### Verbose Mode
Includes all standard data plus:
- Full reasoning chains
- Complete LLM prompt/response pairs
- Detailed debugging information

## Usage Examples

### Basic Logging
```python
from src.logging import DecisionLogger, DecisionType

logger = DecisionLogger("logs/decisions.log")

# Log a signal generation decision
logger.log_decision(
    agent_name="TechnicalAgent",
    agent_type="SignalGenerator",
    decision_type=DecisionType.SIGNAL_GENERATION,
    action_taken="Generated SELL signal",
    confidence_score=0.75,
    reasoning_summary="Bearish divergence detected"
)
```

### Verbose Logging with LLM Data
```python
logger = DecisionLogger("logs/decisions.log", LogLevel.VERBOSE)

logger.log_decision(
    agent_name="AnalysisAgent",
    agent_type="LLMAnalyst",
    decision_type=DecisionType.DATA_ANALYSIS,
    action_taken="Market sentiment: BULLISH",
    confidence_score=0.82,
    reasoning_summary="Positive sentiment outweighs technical concerns",
    full_reasoning="Detailed analysis of multiple factors...",
    llm_prompt="Analyze current market sentiment...",
    llm_response="Based on the data, sentiment is bullish..."
)
```

### Log Analysis
```python
from src.logging import LogAnalyzer

analyzer = LogAnalyzer("logs/decisions.log")

# Read all decisions
decisions = analyzer.read_all_decisions()

# Filter high-confidence decisions
high_confidence = analyzer.filter_decisions(min_confidence=0.8)

# Get summary statistics
summary = analyzer.get_decision_summary()
print(f"Total decisions: {summary['total_decisions']}")
print(f"Average confidence: {summary['confidence_stats']['mean']:.2f}")

# Export to JSON
analyzer.export_to_json("analysis_export.json", include_summary=True)
```

### Agent Integration Pattern
```python
from src.logging import DecisionLogger, DecisionType

class BaseAgent:
    def __init__(self, agent_name, agent_type, decision_logger=None):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.decision_logger = decision_logger or DecisionLogger()
    
    def _log_decision(self, decision_type, action_taken, confidence_score, 
                     reasoning_summary, **kwargs):
        if self.decision_logger:
            return self.decision_logger.log_decision(
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                decision_type=decision_type,
                action_taken=action_taken,
                confidence_score=confidence_score,
                reasoning_summary=reasoning_summary,
                **kwargs
            )

class SignalAgent(BaseAgent):
    def generate_signal(self, market_data):
        # Analysis logic...
        signal = "BUY"
        confidence = 0.85
        
        # Log the decision
        self._log_decision(
            decision_type=DecisionType.SIGNAL_GENERATION,
            action_taken=f"Generated {signal} signal",
            confidence_score=confidence,
            reasoning_summary="Strong trend indicators",
            context_data=market_data
        )
        
        return signal
```

## Log File Format

Decisions are logged as JSON objects, one per line:

```json
{
  "log_level": "standard",
  "event": {
    "agent_name": "SignalAgent_EURUSD",
    "agent_type": "TechnicalAnalysisAgent",
    "timestamp": "2024-01-15T10:30:45.123456",
    "decision_type": "signal_generation",
    "action_taken": "Generated BUY signal for EURUSD",
    "confidence_score": 0.85,
    "reasoning_summary": "Strong bullish indicators detected",
    "full_reasoning": null,
    "llm_prompt": null,
    "llm_response": null,
    "context_data": {
      "symbol": "EURUSD",
      "price": 1.0895,
      "indicators": {"ma_5": 1.089, "rsi": 45.2}
    }
  }
}
```

## Configuration

### Environment Variables
- `AGENT_LOG_LEVEL`: Set to "standard" or "verbose"
- `AGENT_LOG_PATH`: Custom log file path
- `AGENT_LOG_MAX_SIZE_MB`: Maximum log file size before rotation
- `AGENT_LOG_BACKUP_COUNT`: Number of backup files to retain

### Programmatic Configuration
```python
logger = DecisionLogger(
    log_file_path="custom/path/decisions.log",
    log_level=LogLevel.VERBOSE,
    max_file_size_mb=50,  # 50MB before rotation
    backup_count=10       # Keep 10 backup files
)
```

## File Management

The system automatically manages log files through rotation:
- **Automatic Rotation**: When log file exceeds configured size
- **Backup Retention**: Configurable number of backup files
- **Thread Safety**: Safe for concurrent access from multiple agents
- **Directory Creation**: Automatically creates log directories

## Performance Considerations

- **Memory Efficient**: Streaming writes, no in-memory accumulation
- **Thread Safe**: Uses file locking for concurrent agent access
- **Scalable**: File-based storage suitable for MVP requirements
- **Query Performance**: Sequential reads acceptable for typical analysis needs

## Testing

Run the test suite:
```bash
python -m pytest tests/test_logging/ -v
```

The test suite includes:
- 19 comprehensive unit tests
- Edge case handling
- Concurrent access scenarios
- Malformed data handling
- Integration testing

## Demo Scripts

- `demo_decision_logging.py`: Comprehensive demonstration of all features
- `agent_integration_example.py`: Example agent integration patterns

## Integration with Existing Systems

The decision logging system is designed to integrate seamlessly with:
- Existing logging infrastructure (`src/logging_setup.py`)
- Agent orchestration framework (Phase 3)
- Backtesting and analysis workflows
- Performance monitoring and evaluation

## Future Enhancements

Potential improvements for production deployment:
- Database backend for large-scale deployments
- Real-time decision streaming
- Advanced analytics and visualization
- Decision pattern recognition
- Automated decision quality assessment