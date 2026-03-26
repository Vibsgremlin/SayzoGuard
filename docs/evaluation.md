# Evaluation Logs and Outputs

## What was evaluated
- Escrow gating for restricted links
- Regex-based leakage scoring
- Multi-turn context stitching
- LLM-backed leakage classification output format

## Example blocked output
```json
{
  "blocked": true,
  "reason": "escrow_required",
  "message": "Please add money to escrow before sharing restricted links."
}
```

## Example classified output
```json
{
  "blocked": true,
  "rule_score": 3,
  "llm_result": {
    "leakage": true,
    "confidence": 0.91,
    "reason": "Contact detail sharing detected"
  }
}
```

## Notes
- The repo contains the pipeline and dataset-generation code, but no final benchmark report is checked in.
- In-memory session and escrow state mean outputs are demo-oriented rather than production-audited.
