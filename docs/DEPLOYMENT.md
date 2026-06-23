# FinSage — Cloud Deployment Guide

The API (`src/serve/api.py`) serves the **quantized GGUF** model on **CPU** via llama.cpp, so
it runs cheaply almost anywhere. Three tiers below, easiest first.

> Prereq: you have a GGUF file (from `src.merge.merge_and_quantize`) at
> `models/finsage-merged-16bit-gguf/finsage.Q4_K_M.gguf`, or set `FINSAGE_GGUF`.

---

## 0. Run it locally first (sanity check)
```bash
pip install -r requirements-serve.txt
make api                       # uvicorn on :8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the difference between EBITDA and net income?"}'
```
Open the interactive docs at `http://localhost:8000/docs`.

---

## Tier 1 — Docker (portable, works on any VM/PaaS)
```bash
docker compose up --build          # builds image, mounts ./models, serves :8000
```
The model is mounted as a volume (not baked into the image) so the image stays small.
To bake it in for a self-contained image, copy the GGUF in the Dockerfile and remove the volume.

---

## Tier 2 — Managed PaaS (no server to manage)

### Hugging Face Spaces (free tier, great for portfolios)
1. Create a **Docker** Space.
2. Push this repo. Upload the GGUF to the Space (or download it at startup from the HF Hub).
3. Set `app_port: 8000` in the Space's `README.md` front-matter. Done — public URL.

### Render / Railway / Fly.io
- **Render:** New Web Service → Docker → set the GGUF via a persistent disk or download on
  boot → expose port 8000 → health check path `/health`.
- **Fly.io:** `fly launch` (detects the Dockerfile) → `fly volumes create models` for the
  GGUF → `fly deploy`.
- Pick an instance with **≥4GB RAM** for the 3B Q4 model (7B needs ≥8GB).

---

## Tier 3 — Cloud VM (full control)
On an Ubuntu VM (e.g. AWS EC2 `t3.large`, GCP `e2-standard-2`, ~2 vCPU/8GB):
```bash
sudo apt update && sudo apt install -y python3-pip
git clone <your-repo> && cd Finetuning_LLM
pip install -r requirements-serve.txt
# copy the GGUF into ./models, then run under a process manager:
nohup uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 &
```
Put **nginx** in front for TLS, or use `caddy` for automatic HTTPS. Open port 8000 (or 443).

---

## GPU serving (optional, for higher throughput)
For many concurrent users, serve the 16-bit merged model with **vLLM** on a GPU host:
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model <hf-user>/finsage-qwen2.5-3b --max-model-len 4096
```
This exposes an OpenAI-compatible API. Use a GPU PaaS like **Modal**, **RunPod**, or
**Replicate** to avoid managing hardware.

---

## Production hardening checklist (senior expectations)
- [ ] **Auth:** set `api.api_key` (or `FINSAGE_API_KEY`) so `/chat` requires `X-API-Key`.
- [ ] **Rate limiting:** add `slowapi` or do it at the gateway/nginx layer.
- [ ] **CORS:** add `fastapi.middleware.cors` if a browser frontend calls the API.
- [ ] **Observability:** structured logging + `/metrics` (Prometheus) + latency tracking.
- [ ] **Autoscaling:** the app is stateless — scale horizontally behind a load balancer.
- [ ] **Model versioning:** tag GGUF files (`finsage-v1.gguf`) and pin the version per deploy.
- [ ] **Secrets:** keep keys in the platform's secret store / `.env`, never in git.
- [ ] **CI/CD:** extend `.github/workflows/ci.yml` to build + push the Docker image on tag.

---

## Cost sketch (rough, USD)
| Option | Hardware | ~Cost | Good for |
|--------|----------|-------|----------|
| HF Spaces free | shared CPU | $0 | demo / portfolio |
| Small VM / PaaS | 2 vCPU, 8GB | ~$10–25/mo | low traffic |
| GPU PaaS (vLLM) | 1× L4/A10 | per-second billing | bursty / high throughput |
