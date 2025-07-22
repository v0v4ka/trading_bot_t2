# Troubleshooting Guide: Visualization & Agent Issues

## Visualization Issues

### Problem: Chart Not Displaying or Saving
- **Solution:**
  - Ensure output directory exists (`outputs/charts/`).
  - Check file permissions for output path.
  - Verify matplotlib is installed and compatible (`pip install matplotlib`).
  - Use `--output` argument to specify a valid file path.

### Problem: Chart Missing Indicators or Data
- **Solution:**
  - Confirm data source and date range are correct.
  - Check for missing or NaN values in input data.
  - Ensure indicator modules are properly imported and configured.

### Problem: CLI Visualization Command Fails
- **Solution:**
  - Run with `TBOT_TEST_MODE=1` to use synthetic data for debugging.
  - Check CLI arguments for typos or missing required options.
  - Review error messages for missing dependencies.

## Agent Decision Issues

### Problem: No Decisions or Unexpected Actions
- **Solution:**
  - Check agent configuration thresholds in `.env` (e.g., `SIGNAL_CONFIDENCE_THRESHOLD`).
  - Ensure all required signals are present in input data.
  - Review agent logs for error messages or warnings.

### Problem: LLM/DecisionMakerAgent Fails
- **Solution:**
  - Ensure `OPENAI_API_KEY` is set in `.env` or environment.
  - Check API usage limits and network connectivity.
  - Use mock/stub clients for offline testing.

## General Debugging Tips
- Run `pytest` to validate all modules and integration.
- Use verbose logging (`TBOT_LOG_LEVEL=DEBUG`) for more details.
- Review `Memory_Bank.md` for milestone history and troubleshooting context.
