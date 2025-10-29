# Safety Agent

Converting LangGraph_quickstart into custom safety-agent

## Workflow

```
User question
   ↓
Intent detection (is this a statistical question?)
   ↓
If yes → run structured query (e.g., Pandas / SQL / Smartsheet API filter)
If no → normal RAG (semantic retrieval + LLM summarization)
   ↓
Combine the outputs
   ↓
LLM formats the result conversationally
```

## Assumptions

- Pipeline exists to extract Smartsheet data
- Smartsheet data loaded into Azure AI search for the RAG vector store and blob storage csv file for Pandas statistical analysiss
- Start developing with a static query, do not develop front end UI
- Output result in the terminal