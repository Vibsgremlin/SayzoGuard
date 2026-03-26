# Sample Inputs and Outputs

## Sample text input
```json
{
  "text": "Join me on https://meet.google.com/abc-defg-hij",
  "task_id": "task-101",
  "session_id": "session-55"
}
```

## Sample file flow
- Upload a PDF, image, or text file in the Streamlit UI
- Extract text with `extract_text_from_file`
- Send the extracted text to `/classify`

## Sample output after normalization and rule scoring
```json
{
  "blocked": true,
  "rule_score": 3,
  "llm_result": {
    "leakage": true,
    "confidence": 0.88,
    "reason": "Restricted link present"
  }
}
```
