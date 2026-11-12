#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI GPT image model for SPARK steps 5 (per-part images) and 8b (open-state image).

Used by generate_image_part.py / generate_image_open.py: one text prompt + one reference image
-> one PNG, through the Images API *edit* endpoint.

Env knobs (all optional, read at import time):
  SPARK_OPENAI_IMAGE_MODEL    default gpt-image-2
  SPARK_OPENAI_IMAGE_QUALITY  low | medium | high | auto      (default medium)
  SPARK_OPENAI_IMAGE_SIZE     1024x1024 | 1536x1024 | 1024x1536 | auto
                              (default 1024x1024; downstream inference reads link*.png with PIL)
The Images API has no temperature/seed parameters.
"""
import base64
import io
import os
import sys
from pathlib import Path

DEFAULT_MODEL = os.environ.get("SPARK_OPENAI_IMAGE_MODEL", "gpt-image-2").strip()
DEFAULT_QUALITY = os.environ.get("SPARK_OPENAI_IMAGE_QUALITY", "medium").strip()
DEFAULT_SIZE = os.environ.get("SPARK_OPENAI_IMAGE_SIZE", "1024x1024").strip()


def describe_openai_backend(model: str = None, quality: str = None, size: str = None) -> str:
    return f"model={model or DEFAULT_MODEL}, quality={quality or DEFAULT_QUALITY}, size={size or DEFAULT_SIZE}"


def _to_rgb_png(image_bytes: bytes, output_path: str, size: str = None) -> None:
    """Save as RGB PNG (a transparent result is composited onto white, matching the
    'pure white background' the prompts ask for). If the backend ignored the
    requested WxH (the ChatGPT-subscription proxy returns 1254x1254), resize to it so every
    run feeds the inference stage the same resolution."""
    from PIL import Image
    im = Image.open(io.BytesIO(image_bytes))
    if size and "x" in size:
        try:
            w, h = (int(v) for v in size.lower().split("x"))
            if im.size != (w, h):
                im = im.resize((w, h), Image.LANCZOS)
        except ValueError:
            pass
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    im.save(output_path, "PNG")


def generate_image_openai(api_key: str, prompt: str, reference_image_path: str, output_path: str,
                          model: str = None, quality: str = None, size: str = None) -> bool:
    """Generate one image from prompt + reference image with the OpenAI Images API (edit endpoint).

    Returns True on success (PNG written to output_path), False otherwise. Never prints the key.
    """
    model = model or DEFAULT_MODEL
    quality = quality or DEFAULT_QUALITY
    size = size or DEFAULT_SIZE
    try:
        from openai import OpenAI, BadRequestError
    except ImportError:
        print("  ✗ openai package not installed (pip install openai)")
        return False

    client = OpenAI(api_key=api_key)
    kwargs = dict(model=model, prompt=prompt, size=size, quality=quality, n=1)
    # gpt-image-1 / 1.5 accept input_fidelity="high" (keeps the reference's details);
    # gpt-image-2 is always high-fidelity and rejects the parameter, so only send it to 1.x.
    if model.startswith("gpt-image-1"):
        kwargs["input_fidelity"] = "high"

    def _call():
        with open(reference_image_path, "rb") as f:
            return client.images.edit(image=f, **kwargs)

    try:
        try:
            resp = _call()
        except BadRequestError as e:
            if "input_fidelity" in kwargs and "input_fidelity" in str(e):
                kwargs.pop("input_fidelity")
                resp = _call()
            else:
                raise
    except Exception as e:  # auth, quota, moderation, network ... all end the step with exit 1
        print(f"  ✗ OpenAI Images API error ({type(e).__name__}): {str(e)[:500]}")
        return False

    data = getattr(resp, "data", None) or []
    b64 = getattr(data[0], "b64_json", None) if data else None
    if not b64:
        print("  ✗ No image returned from OpenAI Images API")
        return False

    try:
        _to_rgb_png(base64.b64decode(b64), output_path, size)
    except Exception as e:
        print(f"  ✗ Could not decode/save returned image: {e}")
        return False

    usage = getattr(resp, "usage", None)
    if usage is not None:
        print(f"  usage: input_tokens={getattr(usage, 'input_tokens', '?')} "
              f"output_tokens={getattr(usage, 'output_tokens', '?')} ({model}, {quality}, {size})")
    return True


if __name__ == "__main__":  # tiny manual test: python openai_image.py <ref.png> "<prompt>" <out.png>
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    if len(sys.argv) != 4:
        sys.exit("usage: openai_image.py <reference.png> <prompt> <output.png>")
    ok = generate_image_openai(os.environ.get("OPENAI_API_KEY", ""), sys.argv[2], sys.argv[1], sys.argv[3])
    sys.exit(0 if ok else 1)
