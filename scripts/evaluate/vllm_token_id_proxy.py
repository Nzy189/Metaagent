import os
import asyncio
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import aiohttp

from transformers import AutoTokenizer


class TokenizerCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def get(self, model_name: str):
        if model_name not in self._cache:
            if model_name.startswith("models/") and not model_name.startswith("/"):
                actual_model_path = f"/home/lah003/{model_name}"
            else:
                actual_model_path = model_name
            
            tokenizer = AutoTokenizer.from_pretrained(actual_model_path, use_fast=True)
            self._cache[model_name] = tokenizer
        return self._cache[model_name]


def build_app() -> FastAPI:
    app = FastAPI()
    tokenizer_cache = TokenizerCache()

    backend_address = os.environ.get("VLLM_BACKEND_ADDRESS", "127.0.0.1:8101")
    # Handle address that might already contain protocol
    if backend_address.startswith(('http://', 'https://')):
        backend_base = f"{backend_address}/v1"
    else:
        backend_base = f"http://{backend_address}/v1"

    timeout = aiohttp.ClientTimeout(total=2700)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{backend_base}/models") as resp:
                data = await resp.json()
                return JSONResponse(content=data, status_code=resp.status)

    @app.post("/v1/completions")
    async def completions(request: Request):
        req_json = await request.json()

        model_name = req_json.get("model")
        if not model_name:
            return JSONResponse(content={"error": "model is required"}, status_code=400)

        actual_model_name = model_name
        if model_name.startswith("models/") and not model_name.startswith("/"):
            actual_model_name = f"/home/lah003/{model_name}"
        

        req_json_copy = req_json.copy()
        req_json_copy["model"] = actual_model_name

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{backend_base}/completions", json=req_json_copy) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return JSONResponse(content=data, status_code=resp.status)

        try:
            choices = data.get("choices", [])
            if not choices:
                return JSONResponse(content=data)

            tokenizer = tokenizer_cache.get(model_name)

            for choice in choices:
                logprobs = choice.get("logprobs")
                if not logprobs:
                    continue
                tokens = logprobs.get("tokens", [])
                if not tokens:
                    continue

                token_ids = []
                for tok in tokens:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    if isinstance(tid, int) and tid >= 0:
                        token_ids.append(f"token_id:{tid}")
                    else:
                        token_ids.append("token_id:-1")

                logprobs["tokens"] = token_ids

        except Exception:
            pass

        return JSONResponse(content=data)

    return app


def main() -> None:
    host = os.environ.get("PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("PROXY_PORT", "8100"))
    app = build_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()




