---
title: FinSage
emoji: 📈
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FinSage API

FastAPI + llama.cpp serving the fine-tuned **Qwen2.5-3B** GGUF (QLoRA + DPO), CPU inference.

**Endpoint**
```
POST /chat
{ "message": "What is EBITDA?", "history": [["prev user", "prev assistant"]] }
-> { "answer": "..." }
```

CORS is open so the portfolio website can call it from any origin.

Model: [`AshutoshMore96/finsage-gguf`](https://huggingface.co/AshutoshMore96/finsage-gguf)
· Code: [GitHub](https://github.com/AshutoshMore96/finsage-llm-finetuning)
