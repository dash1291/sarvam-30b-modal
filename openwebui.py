import modal

app = modal.App("openwebui")

openwebui_image = modal.Image.from_registry(
    "ghcr.io/open-webui/open-webui:main",
)

WEBUI_PORT = 8080

data_volume = modal.Volume.from_name("openwebui-data", create_if_missing=True)
config_store = modal.Dict.from_name("openwebui-config", create_if_missing=True)


@app.function(
    image=openwebui_image,
    volumes={"/openwebui-storage": data_volume},
    allow_concurrent_inputs=100,
    memory=1024,
)
@modal.web_server(port=WEBUI_PORT, startup_timeout=180)
def serve():
    import subprocess
    import os
    import time
    import socket

    vllm_url = config_store.get("vllm_url", "")
    secret = config_store.get("secret", "change-this-secret")

    env = os.environ.copy()
    env["OPENAI_API_BASE_URL"] = f"{vllm_url}/v1"
    env["OPENAI_API_KEY"] = secret
    env["WEBUI_SECRET_KEY"] = secret
    env["DATA_DIR"] = "/openwebui-storage"
    env["DATABASE_URL"] = "sqlite:////openwebui-storage/webui.db"
    env["UPLOAD_DIR"] = "/openwebui-storage/uploads"
    env["DATABASE_URL"] = "sqlite:////openwebui-storage/webui.db"
    env["PORT"] = str(WEBUI_PORT)
    env["DISABLE_BACKEND_CACHE"] = "1"

    # Make sure storage dir exists
    import os

    os.makedirs("/openwebui-storage", exist_ok=True)
    os.makedirs("/openwebui-storage/uploads", exist_ok=True)

    cmd = [
        "python3",
        "-m",
        "uvicorn",
        "open_webui.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(WEBUI_PORT),
        "--workers",
        "4",
    ]

    print(f"Starting with: {' '.join(cmd)}")

    # Start process in background - don't wait for it
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Give it time to start
    time.sleep(10)

    # Check if it's still running
    if proc.poll() is not None:
        print("Process exited early")
    else:
        print("OpenWebUI started successfully!")


@app.local_entrypoint()
def main(vllm_url: str = "", secret: str = "change-this-secret"):
    import time

    if not vllm_url:
        print("Error: Please provide VLLM_URL")
        print("Usage: python openwebui.py --vllm-url <url> --secret <secret>")
        return

    print(f"Deploying OpenWebUI with vLLM URL: {vllm_url}")

    config_store["vllm_url"] = vllm_url
    config_store["secret"] = secret

    app.deploy()

    from modal import Function

    webui_func = Function.from_name("openwebui", "serve")

    for i in range(30):
        try:
            url = webui_func.get_web_url()
            if url:
                print(f"\n=== OpenWebUI Deployed ===")
                print(f"URL: {url}")
                print(f"Secret: {secret}")
                print(f"\nLogin with email: 'admin@admin.com' and password: '{secret}'")
                break
        except:
            pass
        time.sleep(2)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    vllm_url = ""
    secret = "change-this-secret"

    for i, arg in enumerate(args):
        if arg == "--vllm-url" and i + 1 < len(args):
            vllm_url = args[i + 1]
        if arg == "--secret" and i + 1 < len(args):
            secret = args[i + 1]

    main(vllm_url, secret)
