#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix joint type and axis in metadata.json using rendering.png

This script takes an existing metadata.json and re-evaluates joint types and axes
using GPT-4o vision model, focusing on correcting:
1. Left/right direction detection for revolute joints
2. Distinguishing between revolute (hinged lids) and prismatic (sliding drawers)
"""

import os
import sys
import json
import base64
import argparse
from typing import Any, Dict, List, Optional
from pathlib import Path

# ========= API Configuration =========
# The key is read from the .env file at the repo root, or from the OPENAI_API_KEY env var.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("SPARK_OPENAI_CHAT_MODEL", "gpt-6.1-sol")  # override with the SPARK_OPENAI_CHAT_MODEL env var


def load_image_as_data_uri(path: str) -> str:
    """Load image and convert to base64 data URI."""
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


def create_joint_fix_system_prompt(category: str, parts: List[Dict[str, Any]]) -> str:
    """
    Create system prompt for joint type and axis correction.

    Focus on:
    1. Correctly identifying revolute vs prismatic joints
    2. Accurately determining axis directions (especially left/right)
    """

    # Build parts description for context
    parts_desc = []
    for p in parts:
        label = p.get('label', '')
        name = p.get('name', '')
        parent = p.get('parent', '')
        current_type = p.get('joint_type', '')
        current_axis = p.get('axis', '')

        if current_type == 'fixed':
            parts_desc.append(f"  - {label} ({name}): base part (fixed)")
        else:
            parts_desc.append(f"  - {label} ({name}): parent={parent}, current_type={current_type}, current_axis={current_axis}")

    parts_text = "\n".join(parts_desc)

    return f"""You are a precise mechanical joint analyzer specializing in determining joint types and axes from images.

===== TASK =====
Given an image of a {category} object, you need to verify and correct the joint type and axis for each movable part.

Current parts in the object:
{parts_text}

===== CRITICAL RULES FOR JOINT TYPE =====

1. REVOLUTE vs PRISMATIC - Key Distinctions:

   REVOLUTE (rotating/hinged motion):
   - Doors that swing open (cabinet doors, refrigerator doors, oven doors)
   - Lids that flip up (laptop lids, toilet seats, storage box lids, trash can lids)
   - Covers that rotate open (pots, kettles)
   - Caps that twist (bottle caps)
   - Rotating components (fans, globe rotation)
   - Scissors, pliers (handle rotation)

   PRISMATIC (sliding/linear motion):
   - Drawers that slide out
   - Sliding doors (not swinging doors)
   - Push-pull buttons
   - Extending antennas
   - Sliding keyboards

2. COMMON MISTAKES TO AVOID:
   - ❌ WRONG: Labeling hinged lids/covers as prismatic
   - ✓ CORRECT: Lids that flip open on a hinge are REVOLUTE

   - ❌ WRONG: Labeling drawers as revolute
   - ✓ CORRECT: Drawers that slide in/out are PRISMATIC

   - ❌ WRONG: Labeling cabinet doors as prismatic
   - ✓ CORRECT: Doors that swing on hinges are REVOLUTE

===== CRITICAL RULES FOR AXIS DIRECTION =====

Camera/Image Frame (viewer's perspective):
- +X = RIGHT (→)
- -X = LEFT (←)
- +Y = UP (↑)
- -Y = DOWN (↓)
- +Z = TOWARD CAMERA (front, ⊙)
- -Z = AWAY FROM CAMERA (back, ⊗)

REVOLUTE Joint Axis (rotation axis):
The axis represents the hinge/rotation axis. Think of it as the rod/pin around which the part rotates.

Critical Left/Right Rules:
1. If the hinge is on the LEFT side of the part:
   - The part swings/opens to the RIGHT
   - Rotation axis: "0 1 0" (positive Y-axis pointing UP)
   - OR: "0 -1 0" (negative Y-axis pointing DOWN) - use this if hinge is at bottom-left

2. If the hinge is on the RIGHT side of the part:
   - The part swings/opens to the LEFT
   - Rotation axis: "0 -1 0" (negative Y-axis pointing DOWN)
   - OR: "0 1 0" (positive Y-axis pointing UP) - use this if hinge is at bottom-right

3. If the hinge is on the TOP:
   - The part swings/opens DOWNWARD
   - Rotation axis: "-1 0 0" (negative X-axis pointing LEFT)
   - OR: "1 0 0" (positive X-axis pointing RIGHT) - depends on which side

4. If the hinge is on the BOTTOM:
   - The part swings/opens UPWARD
   - Rotation axis: "1 0 0" (positive X-axis pointing RIGHT)
   - OR: "-1 0 0" (negative X-axis pointing LEFT) - depends on which side

Most Common Cases (use these in 90% of cases):
- Door/lid hinged on LEFT edge: "0 1 0" (opens right)
- Door/lid hinged on RIGHT edge: "0 -1 0" (opens left)
- Lid hinged on TOP edge: "-1 0 0" (opens down)
- Lid hinged on BOTTOM edge: "1 0 0" (opens up)

PRISMATIC Joint Axis (translation direction):
The axis represents the direction of motion in IMAGE coordinates.

Simple mapping:
- Slides RIGHT: "1 0 0"
- Slides LEFT: "-1 0 0"
- Slides UP: "0 1 0"
- Slides DOWN: "0 -1 0"
- Slides FORWARD (toward camera): "0 0 1"
- Slides BACKWARD (away from camera): "0 0 -1"

===== OUTPUT FORMAT =====

Return ONLY a JSON array with corrected joint information for each movable part (skip base/fixed parts):

[
  {{
    "label": "link1",
    "joint_type": "revolute" or "prismatic",
    "axis": "X Y Z",
    "reasoning": "Brief explanation of why this joint_type and axis (1-2 sentences)"
  }},
  ...
]

IMPORTANT:
- axis must be one of: "1 0 0", "-1 0 0", "0 1 0", "0 -1 0", "0 0 1", "0 0 -1"
- For base part (fixed joint), do NOT include it in output
- Focus on visual evidence: where is the hinge? which direction does it move?
- Double-check left/right orientation carefully
- Return ONLY the JSON array, no extra text or markdown
"""


def create_joint_fix_user_prompt(category: str) -> str:
    """Create user prompt for joint correction."""
    return f"""Analyze this {category} image and determine the correct joint_type and axis for each movable part.

Focus on:
1. Is each part revolute (rotating/hinged) or prismatic (sliding/linear)?
2. For revolute: Where is the hinge located? What is the rotation axis?
3. For prismatic: What direction does it slide in?

Return ONLY the JSON array with corrected joint information.
"""


def call_openai_vision(api_key: str, model: str, image_data_uri: str,
                       system_prompt: str, user_prompt: str,
                       seed: Optional[int] = None) -> str:
    """Call OpenAI Chat Completions with vision."""
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
        "temperature": 0.2,
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


def robust_json_extract(text: str) -> Any:
    """Extract JSON from model output."""
    import re

    text = text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Try to parse as JSON array
    if text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Try to find JSON array
    match = re.search(r'\[.*\]', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError(f"Failed to parse JSON from model output: {text[:200]}...")


def fix_joints_for_object(metadata_path: Path, image_path: Path,
                          api_key: str, model: str,
                          seed: Optional[int] = None) -> bool:
    """
    Fix joint types and axes for a single object.

    Args:
        metadata_path: Path to existing metadata.json
        image_path: Path to rendering.png
        api_key: OpenAI API key
        model: Model name
        seed: Random seed for reproducibility

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing metadata
        if not metadata_path.exists():
            print(f"  ✗ metadata.json not found: {metadata_path}")
            return False

        if not image_path.exists():
            print(f"  ✗ Image not found: {image_path}")
            return False

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        category = metadata.get('object_name', 'object')
        parts = metadata.get('parts', [])

        if not parts:
            print(f"  ✗ No parts found in metadata")
            return False

        # Count movable parts (non-fixed)
        movable_parts = [p for p in parts if p.get('joint_type') != 'fixed']
        if not movable_parts:
            print(f"  ℹ No movable parts to fix (all fixed joints)")
            return True

        print(f"  Category: {category}")
        print(f"  Total parts: {len(parts)}, Movable: {len(movable_parts)}")

        # Load image
        data_uri = load_image_as_data_uri(str(image_path))

        # Create prompts
        system_prompt = create_joint_fix_system_prompt(category, parts)
        user_prompt = create_joint_fix_user_prompt(category)

        # Call GPT-4o
        print(f"  Calling {model} to fix joints...")
        raw_response = call_openai_vision(api_key, model, data_uri,
                                         system_prompt, user_prompt, seed)

        # Parse response
        corrections = robust_json_extract(raw_response)

        if not isinstance(corrections, list):
            print(f"  ✗ Invalid response format (expected list)")
            return False

        print(f"  ✓ Received {len(corrections)} corrections")

        # Apply corrections to metadata
        corrections_map = {c['label']: c for c in corrections if 'label' in c}

        changes_made = 0
        for part in parts:
            label = part.get('label')
            if label in corrections_map:
                correction = corrections_map[label]

                old_type = part.get('joint_type')
                old_axis = part.get('axis')

                new_type = correction.get('joint_type')
                new_axis = correction.get('axis')
                reasoning = correction.get('reasoning', '')

                # Apply changes
                if new_type and new_type != old_type:
                    part['joint_type'] = new_type
                    changes_made += 1
                    print(f"    {label}: joint_type {old_type} → {new_type}")

                if new_axis and new_axis != old_axis:
                    part['axis'] = new_axis
                    changes_made += 1
                    print(f"    {label}: axis {old_axis} → {new_axis}")

                if reasoning:
                    print(f"      Reasoning: {reasoning}")

        if changes_made == 0:
            print(f"  ℹ No changes needed (all joints already correct)")
            return True

        # Save updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Applied {changes_made} changes, saved to {metadata_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix joint types and axes in metadata.json using rendering.png"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing subdirectories with metadata.json and rendering.png"
    )
    parser.add_argument(
        "--metadata",
        default="metadata.json",
        help="Metadata filename (default: metadata.json)"
    )
    parser.add_argument(
        "--image",
        default="rendering.png",
        help="Image filename (default: rendering.png)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})"
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
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process input as single directory (not batch mode)"
    )

    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set OPENAI_API_KEY env var.",
              file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Single directory mode
    if args.single or input_path.is_file():
        if input_path.is_file():
            # If input is a file, use its parent directory
            input_path = input_path.parent

        print(f"Processing single directory: {input_path.name}")
        print("-" * 80)

        metadata_path = input_path / args.metadata
        image_path = input_path / args.image

        success = fix_joints_for_object(metadata_path, image_path,
                                       api_key, args.model, args.seed)

        print("-" * 80)
        if success:
            print("✓ Processing completed successfully")
            sys.exit(0)
        else:
            print("✗ Processing failed")
            sys.exit(1)

    # Batch mode - process all subdirectories
    subdirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    if not subdirs:
        print(f"ERROR: No subdirectories found in {input_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Processing {len(subdirs)} subdirectories in {input_path}")
    print("=" * 80)

    success_count = 0
    fail_count = 0
    failed_dirs = []

    for i, subdir in enumerate(subdirs, 1):
        print(f"\n[{i}/{len(subdirs)}] {subdir.name}")
        print("-" * 80)

        metadata_path = subdir / args.metadata
        image_path = subdir / args.image

        if fix_joints_for_object(metadata_path, image_path,
                                api_key, args.model, args.seed):
            success_count += 1
        else:
            fail_count += 1
            failed_dirs.append(subdir.name)

    print("\n" + "=" * 80)
    print(f"Summary: {success_count} succeeded, {fail_count} failed")

    if failed_dirs:
        print("\nFailed directories:")
        for failed_dir in failed_dirs:
            print(f"  - {failed_dir}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
