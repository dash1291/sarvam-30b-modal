import os
import json
import modal


OPENROUTER_MODEL_RESPONSE = {
    "data": [
        {
            "id": "redhatai/sarvam-30b",
            "object": "model",
            "created": 1773928728,
            "name": "RedHatAI: Sarvam-30B",
            "description": "RedHatAI Sarvam-30B is a multilingual LLM supporting Indian languages",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "quantization": "fp8",
            "context_length": 131072,
            "max_output_length": 4096,
            "pricing": {
                "prompt": "0",
                "completion": "0",
                "image": "0",
                "request": "0",
                "input_cache_read": "0",
            },
            "supported_sampling_parameters": [
                "temperature",
                "top_p",
                "max_tokens",
                "stop",
                "presence_penalty",
                "frequency_penalty",
            ],
            "supported_features": ["json_mode", "structured_outputs"],
            "datacenters": [{"country_code": "US"}],
        }
    ]
}

MODEL_NAME = "RedHatAI/sarvam-30b-FP8-Dynamic"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = True

app = modal.App("sarvam-30b")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000

vllm_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:nightly-74fe80ee9594bbc6c0d0c979dbb9d56fae0e789b",
        add_python="3.12",
    )
    .uv_pip_install("requests===2.32.5")
    .entrypoint([])
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

gateway_image = modal.Image.from_registry("python:3.12-slim").pip_install(
    "fastapi>=0.115.0", "uvicorn>=0.30.0", "aiohttp>=0.32.0"
)


vllm_url_store = modal.Dict.from_name("vllm-url-store", create_if_missing=True)


def get_api_token_secret() -> modal.Secret:
    api_token = os.environ.get("API_TOKEN", "")
    if not api_token:
        return modal.Secret.from_name("api-token-store")
    return modal.Secret.from_dict({"API_TOKEN": api_token})


def get_web_app():
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import Response, StreamingResponse
    import aiohttp

    vllm_url = vllm_url_store.get("url", "http://localhost:8000")
    api_token = os.environ.get("API_TOKEN", "")

    web_app = FastAPI()

    async def verify_token(request: Request):
        if not api_token:
            return
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )
        token = auth_header[7:]
        if token != api_token:
            raise HTTPException(status_code=401, detail="Invalid API token")

    @web_app.get("/v1/models")
    async def list_models(request: Request):
        await verify_token(request)
        return OPENROUTER_MODEL_RESPONSE

    @web_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        await verify_token(request)
        body = await request.json()
        model = body.get("model", "")
        stream = body.get("stream", False)

        if model == "redhatai/sarvam-30b":
            body["model"] = "llm"

        timeout = aiohttp.ClientTimeout(total=300)

        if stream:

            async def generate():
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{vllm_url}/v1/chat/completions",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        async for chunk in resp.content.iter_chunked(1024):
                            yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{vllm_url}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    content = await resp.read()
                    return Response(content=content, media_type="application/json")

    @web_app.post("/v1/completions")
    async def completions(request: Request):
        await verify_token(request)
        body = await request.json()
        model = body.get("model", "")
        stream = body.get("stream", False)

        if model == "redhatai/sarvam-30b":
            body["model"] = "llm"

        timeout = aiohttp.ClientTimeout(total=300)

        if stream:

            async def generate():
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{vllm_url}/v1/completions",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        async for chunk in resp.content.iter_chunked(1024):
                            yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{vllm_url}/v1/completions",
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    content = await resp.read()
                    return Response(content=content, media_type="application/json")

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    return web_app


@app.function(image=gateway_image, region=["eu"], secrets=[get_api_token_secret()])
@modal.asgi_app()
def gateway():
    return get_web_app()


@app.cls(
    image=vllm_image,
    region=["eu"],
    gpu=f"L40S:{N_GPU}",
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=32)
class VllmServer:
    @modal.enter(snap=True)
    def start(self):
        import requests
        import socket
        import subprocess

        def wait_ready(proc):
            while True:
                try:
                    socket.create_connection(
                        ("localhost", VLLM_PORT), timeout=1
                    ).close()
                    return
                except OSError:
                    if proc.poll() is not None:
                        raise RuntimeError(f"vLLM exited with {proc.returncode}")

        def warmup():
            payload = {
                "model": "llm",
                "messages": [{"role": "user", "content": "Who are you?"}],
                "max_tokens": 16,
            }
            for ii in range(3):
                requests.post(
                    f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                ).raise_for_status()

        def sleep(level=1):
            requests.post(
                f"http://localhost:{VLLM_PORT}/sleep?level={level}"
            ).raise_for_status()

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--uvicorn-log-level=info",
            "--served-model-name",
            MODEL_NAME,
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--gpu_memory_utilization",
            str(0.95),
        ]

        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)]
        cmd += ["--trust-remote-code"]
        cmd += [
            "--enable-sleep-mode",
            "--max-num-seqs",
            "5",
            "--max-model-len",
            "131072",
            "--max-num-batched-tokens",
            "4096",
            "--enable-chunked-prefill",
        ]
        print(*cmd)

        self.vllm_proc = subprocess.Popen(cmd)
        wait_ready(self.vllm_proc)

        warmup()

        sleep()

    @modal.enter(snap=False)
    def wake_up(self):
        import requests
        import socket

        requests.post(f"http://localhost:{VLLM_PORT}/wake_up").raise_for_status()
        while True:
            try:
                socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
                return
            except OSError:
                pass

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.vllm_proc.terminate()


@app.local_entrypoint()
def main(api_token: str = ""):
    import time

    if api_token:
        os.environ["API_TOKEN"] = api_token

    print("Step 1: Deploying vLLM server...")
    app.deploy()

    print("Waiting for vLLM to be ready and getting URL...")
    vllm_func = modal.Function.from_name("sarvam-30b", "VllmServer.serve")

    for i in range(60):
        try:
            url = vllm_func.get_web_url()
            if url:
                print(f"vLLM URL: {url}")
                break
        except Exception as e:
            print(f"Waiting... ({e})")
        time.sleep(2)
    else:
        print("Could not get vLLM URL")
        return

    print(f"Storing URL in shared dict...")
    vllm_url_store["url"] = url

    print("Redeploying app...")
    app.deploy()

    print("Waiting for gateway to be ready...")
    gateway_func = modal.Function.from_name("sarvam-30b", "gateway")

    for i in range(30):
        try:
            gateway_url = gateway_func.get_web_url()
            if gateway_url:
                print(f"\n=== Deployment Complete ===")
                print(f"Gateway URL: {gateway_url}")
                print(f"Use this URL for OpenRouter")
                if api_token:
                    print(f"API Token: {api_token}")
                break
        except Exception as e:
            print(f"Waiting... ({e})")
        time.sleep(2)


if __name__ == "__main__":
    import sys

    token = sys.argv[1] if len(sys.argv) > 1 else ""
    main(token)
