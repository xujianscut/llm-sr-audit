"""LLM-SR /completions 协议 -> vLLM OpenAI chat 接口的适配器(仅标准库)."""
import json, os, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8011/v1/chat/completions")
MODEL = os.environ["VLLM_MODEL"]
TEMP = float(os.environ.get("LLMSR_TEMP", "0.8"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = dict(model=MODEL, messages=[{"role": "user", "content": req["prompt"]}],
                    n=int(req.get("repeat_prompt", 1)), temperature=TEMP, max_tokens=512)
        r = urllib.request.Request(VLLM, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            out = json.loads(urllib.request.urlopen(r, timeout=600).read())
            content = [c["message"]["content"] for c in out["choices"]]
            code, payload = 200, {"content": content}
        except Exception as e:
            code, payload = 500, {"error": str(e)}
        data = json.dumps(payload).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
