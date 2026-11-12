#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8b: generate the fully-opened state image with the OpenAI GPT image model (see openai_image.py)."""

import os
import sys
import json
import argparse
from pathlib import Path

# ========= API KEY =========
# Read from the .env file at the repo root, or from the OPENAI_API_KEY env var.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ========= Model =========
# OpenAI Images API (gpt-image-2 by default); model/quality/size via SPARK_OPENAI_IMAGE_* (see openai_image.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_image import generate_image_openai, describe_openai_backend  # noqa: E402


def process_open(api_key: str, input_dir: Path) -> bool:
    """
    Generate the fully-opened state image for an articulated object.

    Reads the prompt from prompt_open.json and uses input.png as the
    reference image. Saves the result as open.png in the same directory.

    Args:
        api_key: OpenAI API key
        input_dir: Directory containing prompt_open.json and input.png

    Returns:
        True if successful, False otherwise
    """
    prompt_path = input_dir / "prompt_open.json"
    image_path = input_dir / "input.png"
    output_path = input_dir / "open.png"

    if not prompt_path.exists():
        print(f"  ✗ prompt_open.json not found in {input_dir}")
        return False

    if not image_path.exists():
        print(f"  ✗ input.png not found in {input_dir}")
        return False

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        prompt = prompt_data.get("prompt", "")
        if not prompt:
            print(f"  ✗ 'prompt' field is empty in prompt_open.json")
            return False

        print(f"  Prompt: {prompt}")

        success = generate_image_openai(api_key, prompt, str(image_path), str(output_path))

        if success:
            print(f"  ✓ Saved generated image to: {output_path}")

        return success

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate fully-opened object image using the OpenAI GPT image model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing prompt_open.json and input.png"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API Key (overrides OPENAI_API_KEY)"
    )

    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Generating open-state image for: {input_path}")
    print(f"  Model: {describe_openai_backend()}")
    print("-" * 80)

    success = process_open(api_key, input_path)

    print("-" * 80)
    if success:
        print("Image generation completed successfully")
    else:
        print("Image generation failed")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
