"""VLM backends: in-process transformers (Qwen3-VL, Molmo2) and a tiny HTTP client/server around them.

All backends expose  chat(system, text, images, max_new_tokens) -> (text, stats)  where stats has
latency_s (generate only), tokens_in, tokens_out, model.  Weights come from a local directory (the judged
container has no internet: ship them in the image or mount them under /weights).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.request
from typing import Any

from PIL import Image


def pil_to_b64(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class HFQwen3VL:
    """Qwen3-VL-{8B,32B}-Instruct with SDPA attention in bf16 (or 4-bit via bitsandbytes)."""

    def __init__(self, path: str, *, quant: str = "bf16", device: str = "cuda") -> None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except ImportError:  # older transformers
            from transformers import AutoModelForImageTextToText as Cls
        kw: dict[str, Any] = dict(attn_implementation="sdpa", device_map=device)
        if quant == "bf16":
            kw["dtype"] = torch.bfloat16
        elif quant == "fp8":
            from transformers import FineGrainedFP8Config
            kw["quantization_config"] = FineGrainedFP8Config()
            kw["dtype"] = torch.bfloat16
        elif quant in ("nf4", "4bit"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.torch = torch
        self.model = Cls.from_pretrained(path, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        self.name = f"{os.path.basename(path.rstrip('/'))}:{quant}"

    def chat(self, system: str, text: str, images: list[Image.Image], max_new_tokens: int = 400) -> tuple[str, dict]:
        torch = self.torch
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
        msgs = []
        if system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
        msgs.append({"role": "user", "content": content})
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        gen = out[:, n_in:]
        txt = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return txt, {"latency_s": round(dt, 3), "tokens_in": n_in, "tokens_out": int(gen.shape[1]), "model": self.name}


class HFMolmo:
    """Molmo2-8B pointing: returns pixel points for a text query (used only to ground part descriptions)."""

    def __init__(self, path: str, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, dtype="auto", device_map=device)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, trust_remote_code=True, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
        self.name = os.path.basename(path.rstrip("/"))

    def point(self, image: Image.Image, query: str, max_new_tokens: int = 120) -> tuple[list[tuple[float, float]], dict]:
        torch = self.torch
        w, h = image.size
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"Point to {query}"}, {"type": "image", "image": image}]}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True).to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize(); dt = time.time() - t0
        gen = out[:, n_in:]
        txt = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        pts = parse_molmo_points(txt, w, h)
        return pts, {"latency_s": round(dt, 3), "tokens_in": n_in, "tokens_out": int(gen.shape[1]), "model": self.name, "raw": txt}


def parse_molmo_points(txt: str, w: int, h: int) -> list[tuple[float, float]]:
    """Molmo2 emits <points coords="IMG IDX X Y [IDX X Y ...]">label</points> with X, Y on a 0-1000 grid
    (older Molmo: <point x=".." y=".."> in percent). Returns pixel coordinates."""
    pts: list[tuple[float, float]] = []
    for m in re.finditer(r'<points?\s+coords="([^"]+)"', txt):
        nums = m.group(1).split()
        body = nums[1:]
        for i in range(0, len(body) - 2, 3):
            try:
                pts.append((float(body[i + 1]) / 1000.0 * w, float(body[i + 2]) / 1000.0 * h))
            except ValueError:
                pass
    if not pts:
        for m in re.finditer(r'x(\d*)="([\d.]+)"\s+y\1="([\d.]+)"', txt):
            pts.append((float(m.group(2)) / 100.0 * w, float(m.group(3)) / 100.0 * h))
    return pts


# ----------------------------------------------------------------------------- HTTP wrapper
class RemoteBackend:
    """Client for vlm_server.py (same chat() interface)."""

    def __init__(self, url: str, timeout: float = 120.0) -> None:
        self.url, self.timeout, self.name = url.rstrip("/"), timeout, "remote"

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(self, system, text, images, max_new_tokens=400):
        t0 = time.time()
        r = self._post("/chat", {"system": system, "text": text, "images": [pil_to_b64(i) for i in images],
                                 "max_new_tokens": max_new_tokens})
        st = r["stats"]; st["roundtrip_s"] = round(time.time() - t0, 3); self.name = st.get("model", self.name)
        return r["text"], st

    def point(self, image, query, max_new_tokens=120):
        t0 = time.time()
        r = self._post("/point", {"image": pil_to_b64(image), "query": query, "max_new_tokens": max_new_tokens})
        st = r["stats"]; st["roundtrip_s"] = round(time.time() - t0, 3)
        return [tuple(p) for p in r["points"]], st
