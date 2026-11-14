#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5: generate the isolated image of one part with the OpenAI GPT image model (see openai_image.py)."""

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
# OpenAI Images API (gpt-image-2.6-sunburst by default); model/quality/size via SPARK_OPENAI_IMAGE_* (see openai_image.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_image import generate_image_openai, describe_openai_backend  # noqa: E402


def process_part(api_key: str, input_dir: Path, part_label: str) -> bool:
    """
    Process a single part and generate its image.

    Args:
        api_key: OpenAI API key
        input_dir: Directory containing prompt_part.json and input.png
        part_label: Label of the part to generate (e.g., "link0")

    Returns:
        True if successful, False otherwise
    """
    prompt_path = input_dir / "prompt_part.json"
    image_path = input_dir / "input.png"
    output_path = input_dir / f"{part_label}.png"

    # Check files exist
    if not prompt_path.exists():
        print(f"  ✗ prompt_part.json not found in {input_dir}")
        return False

    if not image_path.exists():
        print(f"  ✗ input.png not found in {input_dir}")
        return False

    try:
        # Load prompts
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        prompts = prompt_data.get("prompts", {})

        if part_label not in prompts:
            print(f"  ✗ Part '{part_label}' not found in prompt_part.json")
            return False

        prompt = prompts[part_label]
        print(f"  Prompt: {prompt}")

        # Generate image
        success = generate_image_openai(api_key, prompt, str(image_path), str(output_path))

        if success:
            print(f"  ✓ Saved generated image to: {output_path}")

        return success

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate part images using the OpenAI GPT image model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing prompt_part.json and input.png"
    )
    parser.add_argument(
        "--part",
        required=True,
        help="Part label to generate image for (e.g., 'link0')"
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

    print(f"Generating image for part: {args.part}")
    print(f"  Model: {describe_openai_backend()}")
    print("-" * 80)

    success = process_part(api_key, input_path, args.part)

    print("-" * 80)
    if success:
        print("Image generation completed successfully")
    else:
        print("Image generation failed")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
