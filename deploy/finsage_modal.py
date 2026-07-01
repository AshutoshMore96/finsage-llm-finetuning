"""
FinSage live demo on Modal — real GPU inference of the fine-tuned model.

Pattern: a lightweight FastAPI web endpoint (CPU) that calls the GPU model class
via .remote(). The GPU container scales to zero when idle (no cost) and spins up
on demand (~20-40s cold start).

Deploy:
    pip install modal
    modal setup                       # one-time browser auth
    modal deploy deploy/finsage_modal.py
    # -> copy the printed  https://<you>--finsage-web.modal.run  URL, add /chat,
    #    put it in index.html's FINSAGE config.
"""
import modal

MODEL = "AshutoshMore96/finsage-merged"   # must be PUBLIC on the HF Hub
SYSTEM = (
    "You are FinSage, a meticulous financial-analysis assistant. Answer the question "
    "directly and concisely. Do not output sentiment labels or extra tokens."
)


def _download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)


# GPU image: torch + transformers + the model weights baked in
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.44", "accelerate", "huggingface_hub")
    .run_function(_download)
)
# tiny CPU image for the web frontend
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi", "pydantic")

app = modal.App("finsage")


@app.cls(gpu="T4", image=gpu_image, scaledown_window=300, timeout=600)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16).to("cuda")
        self.model.eval()

    @modal.method()
    def generate(self, message: str, history: list):
        import torch
        messages = [{"role": "system", "content": SYSTEM}]
        for turn in history or []:
            if len(turn) == 2:
                messages.append({"role": "user", "content": turn[0]})
                messages.append({"role": "assistant", "content": turn[1]})
        messages.append({"role": "user", "content": message})

        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        for junk in ("neutral", "positive", "negative"):
            if text.endswith(junk):
                text = text[: -len(junk)].rstrip()
        return text


@app.function(image=web_image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    api = FastAPI(title="FinSage (Modal)")
    api.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class ChatReq(BaseModel):
        message: str
        history: list = []

    @api.get("/")
    def root():
        return {"service": "FinSage", "status": "ok", "model": MODEL}

    @api.post("/chat")
    def chat(req: ChatReq):
        answer = Model().generate.remote(req.message, req.history)
        return {"answer": answer}

    return api
