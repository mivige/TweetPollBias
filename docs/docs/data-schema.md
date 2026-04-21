# Data Schema

This page outlines the expected JSONL data formats for the raw input files.

## 1. Poll Data (`polls.jsonl`)

The main dataset containing tweet metadata and poll options.

```json
{
  "tweet_id": "123456789987654321",
  "author_id": "12345",
  "created_at": "2026-04-21T10:00:00Z",
  "tweet_text": "If the US presidential election was held today...",
  "poll_options": "[{'position': 1, 'label': 'Trump', 'votes': 1500}, {'position': 2, 'label': 'Harris', 'votes': 1400}]",
  "total_votes": 2900
}
```

## 2. Partisanship & Inference (`Inference/`)

Files in the `Inference/` directory map User IDs to statistical scores or classifications.

### **Partisanship Scores**
One user per line.
```json
{"123456789": 2.11910}
{"987654321": -1.34850}
```

### **Demographics (M3 Inference)**
Nested dictionary containing probabilities for Age, Gender, and Org-status.
```json
{
  "123456789": {
    "gender": {"male": 0.95, "female": 0.05},
    "age": {"<=18": 0.01, "19-29": 0.10, "30-39": 0.40, ">=40": 0.49},
    "org": {"non-org": 0.99, "is-org": 0.01}
  }
}
```

## 3. Engagement Data (`retweeters.jsonl` / `favoriters.jsonl`)

These files map **Poll Tweet IDs** to lists of users who interacted with them. The logic is optimized for memory-safe processing.

```json
{
  "123456789987654321": [
    {"id": "user_id_1"},
    {"id": "user_id_2"}
  ]
}
```
*Note: Each line should contain exactly one Tweet ID key.*
