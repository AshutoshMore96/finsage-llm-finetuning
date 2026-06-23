.PHONY: help install install-mac install-serve data sft dpo merge eval rag compare api test docker

help:
	@echo "FinSage targets:"
	@echo "  install       full training deps (GPU box)"
	@echo "  install-mac   local Mac deps (data prep + MLX inference)"
	@echo "  install-serve API serving deps (FastAPI + llama.cpp)"
	@echo "  data          build SFT dataset"
	@echo "  sft           QLoRA supervised fine-tuning"
	@echo "  dpo           DPO alignment (runs prepare_dpo_data first)"
	@echo "  merge         merge adapters + quantize"
	@echo "  eval          base vs fine-tuned evaluation"
	@echo "  rag           build the RAG FAISS index"
	@echo "  compare       fine-tune vs RAG vs prompting benchmark"
	@echo "  api           run the FastAPI server"
	@echo "  test          run unit tests"
	@echo "  docker        build + run the API container"

install:        ; pip install -r requirements.txt
install-mac:    ; pip install -r requirements-mac.txt
install-serve:  ; pip install -r requirements-serve.txt

data:    ; python -m src.data.prepare_sft_data
sft:     ; python -m src.train.train_sft_qlora
dpo:     ; python -m src.data.prepare_dpo_data && python -m src.train.train_dpo
merge:   ; python -m src.merge.merge_and_quantize
eval:    ; python -m src.eval.evaluate
rag:     ; python -m src.compare.rag_pipeline --build
compare: ; python -m src.compare.benchmark
api:     ; uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
test:    ; pytest -q
docker:  ; docker compose up --build
