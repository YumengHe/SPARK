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

def create_system_prompt(metadata: Dict[str, Any], part_label: str) -> str:
    """
    Create a system prompt for GPT to generate image generation prompt.

    Args:
        metadata: Metadata dictionary containing object and parts info
        part_label: Label of the part to generate prompt for (e.g., "link1")

    Returns:
        System prompt string
    """
    object_name = metadata.get("object_name", "object")
    parts = metadata.get("parts", [])
    num_parts = len(parts)

    # Build list of all parts
    all_parts_list = []
    for part in parts:
        all_parts_list.append(f"{part.get('label', 'unknown')} ({part.get('name', 'unknown')})")

    all_parts_str = ", ".join(all_parts_list)

    # Find the target part
    target_part = None
    target_index = -1
    for i, part in enumerate(parts):
        if part.get("label") == part_label:
            target_part = part
            target_index = i
            break

    if not target_part:
        raise ValueError(f"Part with label '{part_label}' not found in metadata")

    part_name = target_part.get("name", "unknown")

    # Build part names list (without labels)
    part_names_list = [p.get("name", "unknown") for p in parts]
    part_names_str = ", ".join(part_names_list[:-1]) + " and " + part_names_list[-1] if len(part_names_list) > 1 else part_names_list[0] if part_names_list else "unknown"

    return f"""
You are an expert at creating image generation prompts for isolated object parts.

Context:
The reference image shows a {object_name} that is composed of {num_parts} parts: {part_names_str}.

Task:
Based on the reference image, generate an image generation prompt for ONLY the {part_name} (part {target_index + 1} of {num_parts}).

Critical Requirements:
1. The prompt MUST be a single complete sentence that includes background context
2. The prompt should follow this structure:
   - First describe what the original object is and its parts
   - Then state which specific part we want to generate
   - Finally specify: front view, white background, no shadows, centered
3. Describe the texture, material, and color of the part based on the reference image
4. If parts are occluded, imagine the complete shape

Example format:
"This is a {object_name} with {num_parts} parts including {part_names_str}, and we want to generate an isolated image of the {part_name} which has [describe texture/material/color], shown from the front view, centered on a pure white background with no shadows."

Output:
Return ONLY a JSON object with the prompt:
{{"prompt": "<your generated prompt as a single complete sentence>"}}

Do not include explanations or extra text.
"""

def create_user_prompt() -> str:
    """Create user prompt for GPT."""
    return """
Based on the reference image and the part information provided, generate an image generation prompt for this isolated part.

The prompt should describe the part alone on a white background, centered, with no shadows, maintaining the same texture as in the reference image. If any areas are occluded in the reference image, imagine and describe the complete part.

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

def create_aggregate_system_prompt(metadata: Dict[str, Any]) -> str:
    """Create a system prompt for GPT to generate ONE aggregate, English-only prompt
    that instructs an image generator to produce one isolated image per part from
    a single reference image.
    The output of GPT must be a JSON object: {"prompt": "<...>"}.
    """
    object_name = metadata.get("object_name", "object")
    parts = metadata.get("parts", [])
    part_names = [p.get("name", "unknown part") for p in parts]
    joined = ", ".join(part_names) if part_names else "unknown parts"
    n = len(part_names)

    return f"""You are an expert prompt engineer for image generation.
Your goal is to produce ONE concise, high-quality, single-paragraph ENGLISH prompt that
instructs an image-generation model to create {n} isolated images of a "{object_name}",
one image per part. The parts are: {joined}. Each output image contains ONLY the target part,
placed at the exact center, on a plain white background, with no shadows, and with textures/
materials/colors consistent with the reference image. If portions are occluded in the reference,
they must be plausibly completed (inpainted) so the part is fully visible and intact.

Return ONLY a JSON object with the prompt:
{{\"prompt\": \"<your generated prompt in English>\"}}

Do not include any commentary or extra keys."""


def create_aggregate_user_prompt() -> str:
    """User message for the aggregate prompt call. Keep it minimal and English-only."""
    return (
        "Use the provided reference image and the object metadata below to craft ONE single, "
        "concise English prompt that instructs an image-generation model to produce isolated images "
        "for all parts as described. Output ONLY JSON with the 'prompt' field."
    )


def generate_aggregate_prompt(api_key: str, model: str, metadata: Dict[str, Any], image_path: str,
                              seed: Optional[int] = None) -> str:
    """Call GPT to produce the aggregate English prompt based on the same reference image."""
    image_data_uri = load_image_as_data_uri(image_path)
    sys_prompt = create_aggregate_system_prompt(metadata)
    usr_prompt = create_aggregate_user_prompt()
    response_text = call_openai_vision(api_key, model, image_data_uri, sys_prompt, usr_prompt, seed)
    try:
        parsed = json.loads(response_text)
        return parsed.get("prompt", "")
    except Exception:
        return response_text

    """Create user prompt for GPT."""
    return """
Based on the reference image and the part information provided, generate an image generation prompt for this isolated part.

The prompt should describe the part alone on a white background, centered, with no shadows, maintaining the same texture as in the reference image. If any areas are occluded in the reference image, imagine and describe the complete part.

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

def generate_prompts_for_parts(api_key: str, model: str, metadata: Dict[str, Any],
                                image_path: str, seed: Optional[int] = None) -> Dict[str, str]:
    """
    Generate image prompts for all parts using GPT.

    Args:
        api_key: OpenAI API key
        model: Model name to use
        metadata: Metadata dictionary
        image_path: Path to reference image
        seed: Optional random seed

    Returns:
        Dictionary mapping part label to generated prompt
    """
    parts = metadata.get("parts", [])
    prompts = {}

    # Load image once
    print(f"Loading reference image: {image_path}")
    image_data_uri = load_image_as_data_uri(image_path)

    if not parts:
        print("WARNING: No parts found in metadata")
        return prompts

    print(f"Found {len(parts)} part(s) - generating isolated images for each")
    print("-" * 80)

    # Generate prompt for EACH part (including base/fixed parts)
    for i, part in enumerate(parts, 1):
        label = part.get("label", "unknown")
        name = part.get("name", "unknown")

        print(f"[{i}/{len(parts)}] Generating prompt for {label} ({name})...")

        try:
            system_prompt = create_system_prompt(metadata, label)
            user_prompt = create_user_prompt()

            response_text = call_openai_vision(
                api_key, model, image_data_uri,
                system_prompt, user_prompt, seed
            )

            # Parse JSON response
            response_json = json.loads(response_text)
            prompt = response_json.get("prompt", "")

            if prompt:
                prompts[label] = prompt
                print(f"  ✓ Generated prompt: {prompt[:100]}...")
            else:
                print(f"  ✗ Empty prompt returned")

        except Exception as e:
            print(f"  ✗ Error generating prompt: {e}")
            import traceback
            traceback.print_exc()

    return prompts

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

        object_name = metadata.get("object_name", "unknown")
        print(f"  Object: {object_name}")

        # Generate prompts
        prompts = generate_prompts_for_parts(api_key, model, metadata, str(image_path), seed)

        if not prompts:
            print(f"  ✗ No prompts generated")
            return False

        # Save results
        output_data = {
            "object_name": object_name,
            "reference_image": "input.png",
            "prompts": prompts
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Saved {len(prompts)} prompt(s) to: {output_path}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate image generation prompts using GPT for articulated object parts."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input path: directory containing metadata.json and input.png, or parent directory with multiple subdirectories"
    )
    parser.add_argument(
        "--output",
        default="prompt_part.json",
        help="Output filename for prompts (default: prompt_part.json)"
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
