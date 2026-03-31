import time
import urllib.request


def wait_for_local_server(endpoint: str, timeout_sec: int = 600) -> None:
    """Poll vLLM's /health endpoint until ready or timeout."""
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    health_url = f"{base}/health"
    print(f"Waiting for local server at {health_url} (timeout {timeout_sec}s) ...")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as r:
                if r.status == 200:
                    print("Local server is ready.")
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"Local server at {health_url} did not become ready within {timeout_sec}s.")
