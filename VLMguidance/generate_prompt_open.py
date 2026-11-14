#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import base64
import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

# ========= 1) API key =========
# Read from the .env file at the repo root, or from the OPENAI_API_KEY env var.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ========= 2) Default model =========
DEFAULT_MODEL = os.getenv("SPARK_OPENAI_CHAT_MODEL", "gpt-6.1-sol")  # override with the SPARK_OPENAI_CHAT_MODEL env var

# ========= 3) Target opening angle range (in degrees) =========
MIN_OPEN_ANGLE = 120
MAX_OPEN_ANGLE = 180

def load_image_as_data_uri(path: str) -> str:
    """Load image file and convert to data URI for API."""
    mime = "image/jpeg"
    lower = path.lower()
    if lower.endswith(".png"):
        mime = "image/png"
    elif lower.endswith(".gif"):
        mime = "image/gif"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def create_system_prompt(metadata: Dict[str, Any]) -> str:
    """
    Create a system prompt for GPT to generate image generation prompt for fully opened object.

    Args:
        metadata: Metadata dictionary containing object and parts info

    Returns:
        System prompt string
    """
    object_name = metadata.get("object_name", "object")
    parts = metadata.get("parts", [])

    # Find all movable parts
    movable_parts = []
    for part in parts:
        joint_type = part.get("joint_type", "fixed")
        if joint_type != "fixed":
            part_name = part.get("name", "unknown")
            movable_parts.append({
                "name": part_name,
                "joint_type": joint_type,
                "axis": part.get("axis", "0 0 0")
            })

    # Build parts information string
    parts_info = "\n".join([
        f"- {p['name']} ({p['joint_type']} joint, axis: {p['axis']})"
        for p in movable_parts
    ])

    if not movable_parts:
        parts_info = "- No movable parts detected"

    return f"""
You are an expert at creating image generation prompts for articulated objects.

Task:
Given a reference image of a {object_name} and metadata about its movable parts, create a detailed image generation prompt that describes the object with ALL movable parts opened to a WIDE angle (approximately {MIN_OPEN_ANGLE}-{MAX_OPEN_ANGLE} degrees for revolute joints).

Movable parts in this {object_name}:
{parts_info}

Requirements for the prompt:
1. Describe the {object_name} with ALL movable parts opened to a wide angle (approximately {MIN_OPEN_ANGLE}-{MAX_OPEN_ANGLE} degrees for revolute/rotational parts)
2. For prismatic (sliding) joints, describe them as fully extended
3. The opening should be LARGER than typical usage - think of it as demonstrating the full range of motion
4. The prompt should be suitable for an image generation model (like Stable Diffusion or DALL-E)
5. Maintain the same camera angle, lighting, and style as the reference image
6. Be specific about the opened state and how it looks visually
7. Keep the prompt concise but descriptive (2-3 sentences)
8. Focus on the visual appearance of the fully articulated state
9. Do NOT mention technical terms like "joint_type", "axis", "radians", "degrees", etc. Use natural language like "wide open", "fully extended", "spread wide"

Camera/Image axes convention (for your understanding):
- +x: right
- +y: up
- +z: front (toward camera)

Example: For a microwave, instead of saying "door open 90 degrees", say "door wide open, revealing the full interior".

Output:
Return ONLY a JSON object with the prompt:
{{"prompt": "<your generated prompt>"}}

Do not include explanations or extra text.
"""

def create_user_prompt() -> str:
    """Create user prompt for GPT."""
    return """
Based on the reference image and the parts information provided, generate an image generation prompt that describes this object with all movable parts opened wide to demonstrate their full range of motion (approximately 120-180 degrees for rotational parts).

Return only the JSON with the prompt.
"""

def call_openai_vision(api_key: str, model: str, image_data_uri: str,
                       system_prompt: str, user_prompt: str,
                       seed: Optional[int] = None) -> str:
    """
    Call OpenAI Chat Completions with a text+image message.
    Returns the raw text content from the assistant.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "Missing `openai` package. Install with: pip install --upgrade openai"
        ) from e

    client = OpenAI(api_key=api_key)

    user_content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]

    # Build API call parameters
    api_params = {
        "model": model,
        "temperature": 0.7,  # Higher temperature for more creative prompts
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    # Add seed if provided
    if seed is not None:
        api_params["seed"] = seed

    resp = client.chat.completions.create(**api_params)
    return resp.choices[0].message.content

def generate_open_prompt(api_key: str, model: str, metadata: Dict[str, Any],
                         image_path: str, seed: Optional[int] = None) -> str:
    """
    Generate image prompt for fully opened object using GPT.

    Args:
        api_key: OpenAI API key
        model: Model name to use
        metadata: Metadata dictionary
        image_path: Path to reference image
        seed: Optional random seed

    Returns:
        Generated prompt string
    """
    print(f"  Loading reference image: {image_path}")
    image_data_uri = load_image_as_data_uri(image_path)

    object_name = metadata.get("object_name", "unknown")
    print(f"  Object: {object_name}")

    # Check for movable parts
    parts = metadata.get("parts", [])
    movable_count = sum(1 for p in parts if p.get("joint_type") != "fixed")

    if movable_count == 0:
        print("  WARNING: No movable parts found in metadata")
        return ""

    print(f"  Found {movable_count} movable part(s)")
    print("  Generating prompt for fully opened state...")

    try:
        system_prompt = create_system_prompt(metadata)
        user_prompt = create_user_prompt()

        response_text = call_openai_vision(
            api_key, model, image_data_uri,
            system_prompt, user_prompt, seed
        )

        # Parse JSON response
        response_json = json.loads(response_text)
        prompt = response_json.get("prompt", "")

        if prompt:
            print(f"  ✓ Generated prompt: {prompt[:100]}...")
            return prompt
        else:
            print(f"  ✗ Empty prompt returned")
            return ""

    except Exception as e:
        print(f"  ✗ Error generating prompt: {e}")
        import traceback
        traceback.print_exc()
        return ""

def process_single_directory(api_key: str, model: str, input_dir: Path,
                             output_filename: str, seed: Optional[int] = None) -> bool:
    """
    Process a single directory containing metadata.json and input.png.

    Returns:
        True if successful, False otherwise
    """
    metadata_path = input_dir / "metadata.json"
    image_path = input_dir / "input.png"
    output_path = input_dir / output_filename

    # Check files exist
    if not metadata_path.exists():
        print(f"  ✗ metadata.json not found in {input_dir}")
        return False

    if not image_path.exists():
        print(f"  ✗ input.png not found in {input_dir}")
        return False

    try:
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Generate prompt
        prompt = generate_open_prompt(api_key, model, metadata, str(image_path), seed)

        if not prompt:
            print(f"  ✗ No prompt generated")
            return False

        # Save result
        output_data = {
            "object_name": metadata.get("object_name", "unknown"),
            "reference_image": "input.png",
            "prompt": prompt,
            "description": f"All movable parts opened to wide angles ({MIN_OPEN_ANGLE}-{MAX_OPEN_ANGLE} degrees)"
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Saved prompt to: {output_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate image generation prompts using GPT for fully opened articulated objects."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input path: directory containing metadata.json and input.png, or parent directory with multiple subdirectories"
    )
    parser.add_argument(
        "--output",
        default="prompt_open.json",
        help="Output filename for prompt (default: prompt_open.json)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API Key (overrides OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (optional)"
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

    # Check if input is a single directory with metadata.json or parent directory
    metadata_in_input = (input_path / "metadata.json").exists()

    if metadata_in_input:
        # Single directory mode
        print(f"Processing directory: {input_path.name}")
        print("-" * 80)

        success = process_single_directory(api_key, args.model, input_path, args.output, args.seed)

        print("-" * 80)
        if success:
            print("Processing completed successfully")
        else:
            print("Processing failed")
        sys.exit(0 if success else 1)

    else:
        # Multiple directories mode
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]

        if not subdirs:
            print(f"ERROR: No subdirectories found in {input_path}", file=sys.stderr)
            sys.exit(2)

        print(f"Found {len(subdirs)} subdirectories in {input_path}")
        print("-" * 80)

        success_count = 0
        fail_count = 0
        failed_dirs = []

        for i, subdir in enumerate(sorted(subdirs), 1):
            print(f"[{i}/{len(subdirs)}] Processing {subdir.name}")

            if process_single_directory(api_key, args.model, subdir, args.output, args.seed):
                success_count += 1
            else:
                fail_count += 1
                failed_dirs.append(subdir.name)

            print()

        print("-" * 80)
        print(f"Summary: {success_count} succeeded, {fail_count} failed")

        if failed_dirs:
            print("\nFailed directories:")
            for failed_dir in failed_dirs:
                print(f"  - {failed_dir}")

if __name__ == "__main__":
    main()
