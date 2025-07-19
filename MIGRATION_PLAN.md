# Configuration Migration Plan

## Problem
- Duplicate settings between `config/llm_config.yml` and `.env`
- Two sources of truth for the same configuration values
- `config_dir` variable no longer needed with .env approach

## Current Overlaps
| Setting | YAML Location | ENV Variable |
|---------|---------------|--------------|
| LLM Model | `agents.llm_model` | `OPENAI_MODEL` |
| Signal Threshold | `agents.signal_confidence_threshold` | `SIGNAL_CONFIDENCE_THRESHOLD` |
| Decision Threshold | `agents.decision_confidence_threshold` | `DECISION_CONFIDENCE_THRESHOLD` |
| Initial Capital | `backtest.initial_capital` | `DEFAULT_INITIAL_CAPITAL` |
| LLM Confirmation | `agents.use_llm_confirmation` | `USE_LLM_CONFIRMATION` |

## Recommendation: Option A (Gradual Migration)

### Phase 1: Deprecate Overlaps ✅
- [x] Remove `CONFIG_DIR` from env_config.py
- [x] Remove `CONFIG_DIR` from .env files
- [ ] Update `src/llm_config.py` to prefer .env over YAML
- [ ] Add deprecation warnings for YAML usage

### Phase 2: Keep config/ for Complex Scenarios
- Keep `config/` for **multi-environment setups** (dev/staging/prod)
- Keep `config/` for **complex backtesting scenarios**
- Keep `config/` for **agent orchestration workflows**

### Phase 3: Clean Migration
- [ ] Move simple settings to .env only
- [ ] Use config/ only for structured data that benefits from YAML
- [ ] Update documentation to clarify the split

## Alternative: Option B (Full Migration)

### Remove config/ entirely
- [ ] Move all YAML settings to .env
- [ ] Update all hardcoded "config/llm_config.yml" references
- [ ] Remove config directory
- [ ] Update .vscode/launch.json debug configs

## Files to Update
- `src/llm_config.py` - Make .env precedence over YAML
- `.vscode/launch.json` - Update debug configs
- Documentation - Clarify configuration approach

## Testing
- [x] Verify test_env.py still works after config_dir removal
- [ ] Verify all CLI commands work with .env only
- [ ] Verify VS Code debug configurations still work
