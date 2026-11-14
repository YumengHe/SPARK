#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import base64
import argparse
import re
import csv
from pathlib import Path
from typing import Any, Dict, Optional

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

CATEGORIES = [
    "Bottle","Box","Bucket","Camera","Cart","Chair","Clock","CoffeeMachine","Dishwasher","Dispenser","Display",
    "Door","Eyeglasses","Fan","Faucet","FoldingChair","Globe","Kettle","Keyboard","KitchenPot","Knife","Lamp",
    "Laptop","Lighter","Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Refrigerator","Remote",
    "Safe","Scissors","Stapler","StorageFurniture","Suitcase","Switch","Table","Toaster","Toilet","TrashCan","USB",
    "WashingMachine","Window"
]

def load_category_info(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load category information from CSV file.
    Returns a dict mapping category name to its info (min_parts, max_parts, avg_parts, part_names).
    """
    category_info = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = row['Category']
                # Parse part names from string representation of list
                part_names_str = row['Part Names']
                try:
                    # Use ast.literal_eval for safe parsing
                    import ast
                    part_names = ast.literal_eval(part_names_str)
                except:
                    part_names = []

                category_info[category] = {
                    'min_parts': float(row['Min Parts']),
                    'max_parts': float(row['Max Parts']),
                    'avg_parts': float(row['Average Parts']),
                    'part_names': part_names
                }
        print(f"Loaded info for {len(category_info)} categories from {csv_path}")
    except Exception as e:
        print(f"Warning: Failed to load CSV: {e}")

    return category_info

# ========= STEP 1: Category prediction prompt =========
def create_category_system_prompt(hint: Optional[str] = None) -> str:
    """
    Create a system prompt for category prediction.
    """
    return f"""
    You are a vision-capable classifier.

    ===== HIGH-PRIORITY HINT (MAY BE IN ANY LANGUAGE) =====
    {hint}
    =======================================================

    HINT WINS POLICY:
    - If the HINT explicitly states or clearly implies the object's category (even indirectly or in another language), you MUST use that category.
    - If the HINT's category is not exactly one of the allowed labels, map it to the closest allowed label by meaning. HINT takes precedence over the image and priors.

    Task:
    Choose the object's category strictly from this allowed list:
    {CATEGORIES}

    Output Constraints:
    - Return ONLY a JSON object: {{"category": "<one of the allowed labels>"}}.
    - No explanations, no commentary, no markdown.

    Self-check before you answer:
    - Did I follow the HINT over the image if they conflict?
    - Is the category one of the allowed labels?

    """

CATEGORY_USER_PROMPT = """
Identify the object's category from the image, applying the HINT WINS POLICY.
Return ONLY: {{"category":"..."}}.
"""

# ========= STEP 2: Parts prediction prompt =========
TEMPLATE_HINT = """{
    "object_name": "<ONE OF THE LISTED CATEGORIES>",
    "num_parts": <INTEGER>,
    "parts": [
        {
            "label": "link0",
            "name": "<base_name>",
            "parent": "base",
            "joint_type": "fixed",
            "axis": "0 0 0",
            "limit_lower": "0",
            "limit_upper": "0"
        },
        {
            "label": "link1",
            "name": "<part_name>",
            "parent": "link0",
            "joint_type": "revolute|prismatic|fixed",
            "axis": "1 0 0|0 1 0|0 0 1|-1 0 0|0 -1 0|0 0 -1|0 0 0",
            "limit_lower": "<deg_or_m>",
            "limit_upper": "<deg_or_m>"
        }
    ]
}"""

def create_parts_system_prompt(category: str, category_info: Dict[str, Any], hint: Optional[str] = None) -> str:
    """
    Create a system prompt for parts prediction based on category information.
    """
    min_parts = int(category_info['min_parts'])
    max_parts = int(category_info['max_parts'])
    avg_parts = category_info['avg_parts']
    part_names = category_info['part_names']

    part_names_str = ", ".join(part_names) if part_names else "common part names"

    # Build hint section if hint is provided
    return f"""
You are a precise mechanical annotator that outputs STRICT JSON ONLY.

===== HIGH-PRIORITY HINT (MAY BE IN ANY LANGUAGE) =====
{hint}
=======================================================

HINT WINS POLICY:
- If the HINT specifies or implies the number of parts, the base name, or any part names/joint types, you MUST follow the HINT exactly.
- When the HINT and the image/prior disagree, obey the HINT.

Task:
From a single image of a {category}, identify kinematic parts and joints.

Category Information:
- Object category: {category}
- Typical number of parts: {avg_parts:.1f} (range: {min_parts}-{max_parts})
- Common part names for this category: {part_names_str}

Instructions:
(1) Identify ONLY kinematic parts (parts that move). Choose a base link that represents the static/main body.
(2) IMPORTANT: Only the base part (link0) can have joint_type "fixed". All other parts MUST be kinematic (movable) with joint_type "revolute" or "prismatic".
(3) If a component does not move relative to the base, it should NOT be listed as a separate part. Only include parts that have motion.
(4) IMPORTANT: The common part names listed above are ALL possible parts for this {category} category. From this list, select ONLY the parts that are VISIBLE and PRESENT in the given image.
(5) IMPORTANT: The same part can appear MULTIPLE times. For example, if there are multiple drawers, include them as separate entries with the same part name (e.g., "drawer", "drawer", "drawer"). Each instance should have its own link label (link1, link2, link3, etc.).
(6) For each kinematic part, infer joint_type (revolute | prismatic), axis, and motion limits.
(7) The number of parts should be between {min_parts} and {max_parts}.

Conventions:
- Camera/image axes: +x right, +y up, +z front (toward camera).

- Revolute joint axis:
  IMPORTANT: The axis represents the rotation axis. Based on this axis, clockwise is negative and counter-clockwise is positive.
  Most common revolute joint axes (use these in most cases):
  - Opening/rotating LEFT (hinge on left side): "0 -1 0"
  - Opening/rotating RIGHT (hinge on right side): "0 1 0"
  - Opening/rotating UP (hinge on top): "1 0 0"
  - Opening/rotating DOWN (hinge on bottom): "-1 0 0"
  Less common (rarely used):
  - Forward rotation: "0 0 1"
  - Backward rotation: "0 0 -1"

- Prismatic joint axis uses IMAGE directions (screen-aligned), not the object's local frame. Map directions to numeric vectors and OUTPUT the numeric string:
  - front → "0 0 1"
  - back  → "0 0 -1"
  - up    → "0 1 0"
  - down  → "0 -1 0"
  - right → "1 0 0"
  - left  → "-1 0 0"

- All axes must be one of these unit vectors (or "0 0 0" for fixed):
  [1 0 0], [0 1 0], [0 0 1], [-1 0 0], [0 -1 0], [0 0 -1].

Limits:
- Revolute: use radians (decimal). Set "limit_lower" = "0" to represent the image pose as zero; choose a plausible positive "limit_upper" (e.g., "1.5707963267948966").
- Prismatic: use meters (decimal). Set "limit_lower" = "0"; choose a positive "limit_upper" translation distance.
- Fixed: both "0" (ONLY for base part).

Output Schema (STRICT JSON, no extra keys, no commentary):
{{
  "object_name": "{category}",
  "num_parts": <int>,
  "parts": [
    {{
      "label": "link0",
      "name": "<base-name>",
      "parent": "base",
      "joint_type": "fixed",
      "axis": "0 0 0",
      "limit_lower": "0",
      "limit_upper": "0"
    }},
    {{
      "label": "link1",
      "name": "<part-name>",
      "parent": "link0",
      "joint_type": "revolute" | "prismatic",
      "axis": "<one of the allowed axes above>",
      "limit_lower": "0",
      "limit_upper": "<positive number>"
    }}
    // ... more links ...
  ]
}}

Validation Checklist (perform mentally BEFORE answering; if any item fails, revise your JSON FIRST):
- [HINT] Have I exactly followed the HINT's part names/count/base/joint types where specified?
- [BASE] Exactly one base (link0, fixed, parent 'base', axis '0 0 0').
- [COUNT] num_parts == length of parts.
- [KINEMATIC] No non-base part has joint_type "fixed".
- [AXIS] Every axis is one of the allowed unit directions in IMAGE frame.
- [LIMITS] All limit_lower == "0"; all non-base parts have positive limit_upper.

Return ONLY the final JSON. No explanations, no markdown.
"""

PARTS_USER_PROMPT = """
Infer kinematic parts for the {PREDICTED_CATEGORY} image, obeying the HINT WINS POLICY.
Return STRICT JSON only following the schema; no extra text.
"""

def load_image_as_data_uri(path: str) -> str:
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

def robust_json_extract(text: str) -> Dict[str, Any]:
    """
    Try to extract a top-level JSON object from the model text.
    """
    # If it's already pure JSON:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Fallback: find the first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    raise ValueError("Failed to parse JSON from model output.")

def call_openai_vision(api_key: str, model: str, image_data_uri: str, system_prompt: str, user_prompt: str, seed: Optional[int] = None) -> str:
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
        "temperature": 0.2,
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
    # Extract first choice content
    return resp.choices[0].message.content

def process_single_image(api_key: str, model: str, image_path: str, output_path: str,
                         category_info_dict: Dict[str, Dict[str, Any]], seed: Optional[int] = None, hint: Optional[str] = None) -> bool:
    """
    Process a single image and generate JSON metadata using two-step prediction.

    Step 1: Predict category
    Step 2: Predict parts based on category information
    """
    try:
        if not os.path.exists(image_path):
            print(f"  ✗ Image not found: {image_path}")
            return False

        data_uri = load_image_as_data_uri(image_path)

        if hint:
            print(f"  Found hint: {hint[:100]}..." if len(hint) > 100 else f"  Found hint: {hint}")

        # ========= Step 1: Predict category =========
        print("  Step 1: Predicting category...")
        category_system_prompt = create_category_system_prompt(hint)
        category_raw_text = call_openai_vision(api_key, model, data_uri, category_system_prompt, CATEGORY_USER_PROMPT, seed)
        category_result = robust_json_extract(category_raw_text)

        predicted_category = category_result.get("category", "")

        # Normalize category name
        if predicted_category not in CATEGORIES:
            # Try best-effort mapping: case-insensitive match
            cand = str(predicted_category).strip()
            for c in CATEGORIES:
                if c.lower() == cand.lower():
                    predicted_category = c
                    break

        if predicted_category not in CATEGORIES:
            print(f"  ✗ Invalid category predicted: {predicted_category}")
            return False

        print(f"  ✓ Predicted category: {predicted_category}")

        # Get category information
        if predicted_category not in category_info_dict:
            print(f"  ✗ No category info found for: {predicted_category}")
            return False

        category_info = category_info_dict[predicted_category]
        print(f"    Expected parts: {category_info['avg_parts']:.1f} (range: {int(category_info['min_parts'])}-{int(category_info['max_parts'])})")

        # ========= Step 2: Predict parts based on category info =========
        print("  Step 2: Predicting parts and joints...")
        parts_system_prompt = create_parts_system_prompt(predicted_category, category_info, hint)
        parts_user_prompt = f"""
Infer kinematic parts for the {predicted_category} image, obeying the HINT WINS POLICY.
Return STRICT JSON only following the schema; no extra text.
"""
        parts_raw_text = call_openai_vision(api_key, model, data_uri, parts_system_prompt, parts_user_prompt, seed)
        result = robust_json_extract(parts_raw_text)

        # Ensure object_name matches predicted category
        result["object_name"] = predicted_category

        # Ensure num_parts matches
        if isinstance(result.get("parts"), list):
            result["num_parts"] = len(result["parts"])
            print(f"  ✓ Generated {result['num_parts']} parts")

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved: {output_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Infer articulated parts and joints from images using OpenAI GPT.")
    parser.add_argument("--input", required=True, help="Input path: either a directory containing subfolders with images, or a single image file")
    parser.add_argument("--csv", default="partnet-mobility-data-analysis.csv", help="CSV file with category information (default: partnet-mobility-data-analysis.csv)")
    parser.add_argument("--image", default="rendering.png", help="Image filename to look for in each subfolder when --input is a directory (default: rendering.png)")
    parser.add_argument("--output", default=None, help="Output JSON file path (only used when --input is a single image)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=None, help="OpenAI API Key (overrides OPENAI_API_KEY/env).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results (optional)")
    args = parser.parse_args()

    # Resolve API key precedence: CLI > module var > env
    api_key = args.api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set OPENAI_API_KEY env var or fill OPENAI_API_KEY in script.", file=sys.stderr)
        sys.exit(2)

    from pathlib import Path
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Load category information from CSV
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    category_info_dict = load_category_info(csv_path)
    if not category_info_dict:
        print(f"ERROR: Failed to load category information from {csv_path}", file=sys.stderr)
        sys.exit(2)

    # Check if input is a single image file or a directory
    if input_path.is_file():
        # Single image mode
        print(f"Processing single image: {input_path.name}")
        print("-" * 80)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / "metadata.json"

        # Check for hint.txt in the same directory
        hint_path = input_path.parent / "hint.txt"
        hint = None
        if hint_path.exists():
            try:
                with open(hint_path, 'r', encoding='utf-8') as f:
                    hint = f.read().strip()
            except Exception as e:
                print(f"  Warning: Failed to read hint.txt: {e}")

        success = process_single_image(api_key, args.model, str(input_path), str(output_path), category_info_dict, args.seed, hint)

        print("-" * 80)
        if success:
            print("Processing completed successfully")
        else:
            print("Processing failed")
        sys.exit(0 if success else 1)

    else:
        # Directory mode - process all subdirectories
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

            image_path = subdir / args.image
            output_path = subdir / "metadata.json"

            # Check for hint.txt in the subfolder
            hint_path = subdir / "hint.txt"
            hint = None
            if hint_path.exists():
                try:
                    with open(hint_path, 'r', encoding='utf-8') as f:
                        hint = f.read().strip()
                except Exception as e:
                    print(f"  Warning: Failed to read hint.txt: {e}")

            if process_single_image(api_key, args.model, str(image_path), str(output_path), category_info_dict, args.seed, hint):
                success_count += 1
            else:
                fail_count += 1
                failed_dirs.append(subdir.name)

        print("-" * 80)
        print(f"Summary: {success_count} succeeded, {fail_count} failed")

        if failed_dirs:
            print("\nFailed directories:")
            for failed_dir in failed_dirs:
                print(f"  - {failed_dir}")

if __name__ == "__main__":
    main()
