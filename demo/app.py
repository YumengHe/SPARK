from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# API keys come from the .env file at the repo root (or the environment).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


OUTPUT_ROOT = PROJECT_ROOT / "output"
INPUT_ROOT = PROJECT_ROOT / "image"
CSV_PATH = PROJECT_ROOT / "VLMguidance" / "partnet-mobility-data-analysis.csv"
EXAMPLE_IMAGE = INPUT_ROOT / "45162.png"
EXAMPLE_RUN_DIR = OUTPUT_ROOT / "45162_20260501_230933"
GRADIO_TMP_ROOT = PROJECT_ROOT / "demo" / "tmp"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_INFERENCE_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_CUDA_DEVICE = "0"

GRADIO_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TMP_ROOT))
os.environ.setdefault("TMPDIR", str(GRADIO_TMP_ROOT))

import gradio as gr
from PIL import Image


HEADER = """
# SPARK Demo
"""

HERO_HTML = """
<div class="hero-wrap">
    <div class="hero-title">SPARK: Sim-ready Part-level Articulated Reconstruction</div>
    <p class="hero-subtitle">
        Generate articulated object structure, part images, and PartCrafter 3D results from a single image.
    </p>

    <div class="hero-instructions">
        <h3>How to Use</h3>
        <p><strong>🚀 Quick Start:</strong> Click <strong>▶ Run Example</strong> to preview the full UI.</p>
        <p><strong>📋 Custom Workflow:</strong></p>
        <ol>
            <li><strong>Upload Image</strong> - Select your input image</li>
            <li><strong>Fill OpenAI Key</strong> - Required for the full pipeline (metadata, prompts, and image generation)</li>
            <li><strong>Adjust Generation Controls</strong> - Tune seed, steps, and CFG</li>
            <li><strong>Click Generate 3D Model</strong> - Generate metadata, URDF, part images, and 3D result</li>
        </ol>
    </div>
</div>
"""

CUSTOM_CSS = """
.gradio-container {
    max-width: 1500px !important;
    background: #f8fafc;
}

.hero-wrap {
    padding: 18px 22px;
    border-radius: 18px;
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    margin-bottom: 12px;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    margin: 0 0 14px 0;
    color: #334155;
}

.hero-instructions {
    background: #f8fafc;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    padding: 14px 16px;
}

.hero-instructions h3 {
    margin: 0 0 10px 0;
    font-size: 1.05rem;
}

.hero-instructions p,
.hero-instructions ol {
    margin: 0 0 8px 0;
    color: #334155;
}

.panel-card {
    border: 1px solid rgba(148, 163, 184, 0.24);
    background: rgba(255, 255, 255, 0.96);
    border-radius: 18px;
    padding: 18px !important;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
}

.section-title {
    margin-bottom: 8px;
}

.section-title h2 {
    margin: 0;
    font-size: 1.25rem;
}

.section-title p {
    margin: 6px 0 0 0;
    color: #475569;
    font-size: 0.95rem;
}

.status-card {
    border-radius: 14px;
    padding: 12px 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.hint-text {
    color: #475569;
    font-size: 0.92rem;
}

.subsection-title {
    margin: 10px 0 6px 0;
    font-size: 1rem;
    font-weight: 600;
    color: #0f172a;
}

.results-top-row {
    gap: 12px;
}

.gradio-container .gr-button-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
}

.compact-tabs {
    margin-top: 8px;
}

.examples-panel {
    margin-top: 10px;
}

.examples-strip-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 10px 12px;
}

.examples-strip-title {
    margin-bottom: 6px;
}

.examples-strip-title h3 {
    margin: 0;
    font-size: 0.95rem;
}

.examples-strip-title p {
    margin: 2px 0 0 0;
    color: #64748b;
    font-size: 0.84rem;
}
"""


class DemoError(RuntimeError):
    pass


def run_command(command: list[str], env: dict[str, str], title: str) -> str:
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    log = [f"\n===== {title} =====", "$ " + " ".join(command)]
    if result.stdout:
        log.append(result.stdout.strip())
    if result.stderr:
        log.append("[stderr]\n" + result.stderr.strip())
    if result.returncode != 0:
        raise DemoError("\n".join(log))
    return "\n".join(log)


def prepare_env(openai_api_key: str, cuda_device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = cuda_device or DEFAULT_CUDA_DEVICE
    if openai_api_key:
        env["OPENAI_API_KEY"] = openai_api_key
    return env


def ensure_api_keys(openai_api_key: str) -> None:
    if not (openai_api_key or os.getenv("OPENAI_API_KEY")):
        raise DemoError(
            "The SPARK pipeline requires an **OpenAI API key**.\n\n"
            "- `OpenAI API Key`: used for `metadata.json` prediction, joint-axis correction, prompt generation, "
            "and part-image / open-state image generation (gpt-image-2)\n\n"
            "Please fill in the field under `Model & API Settings`, or set `OPENAI_API_KEY` in your environment."
        )


def create_run_dir(image_path: str) -> Path:
    stem = Path(image_path).stem or "spark"
    run_dir = OUTPUT_ROOT / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_input_image(image_path: str, run_dir: Path) -> Path:
    output_path = run_dir / "input.png"
    image = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(bg, image).convert("RGB")
    merged.save(output_path)
    return output_path


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def list_part_images(run_dir: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for path in sorted(run_dir.glob("link*.png")):
        items.append((str(path), path.stem))
    return items


def find_inference_dir(run_dir: Path) -> Path | None:
    inference_root = run_dir / "inference"
    if not inference_root.exists():
        return None
    candidates = sorted([p for p in inference_root.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None


def choose_render_image(run_dir: Path) -> str | None:
    inference_dir = find_inference_dir(run_dir)
    if not inference_dir:
        return None
    for name in ("rendering_angle.png", "rendering.png", "rendering_grid.png"):
        candidate = inference_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def choose_model_path(run_dir: Path) -> str | None:
    inference_dir = find_inference_dir(run_dir)
    if not inference_dir:
        return None
    for name in ("object.glb", "part_00_cleaned.glb", "part_00.glb"):
        candidate = inference_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def build_archive(run_dir: Path) -> str | None:
    archive_base = str(run_dir)
    archive_path = archive_base + ".zip"
    if Path(archive_path).exists():
        return archive_path
    shutil.make_archive(archive_base, "zip", root_dir=run_dir)
    return archive_path


def summarize_run(run_dir: Path) -> str:
    metadata = read_json_if_exists(run_dir / "metadata.json") or {}
    num_parts = metadata.get("num_parts", "?")
    object_name = metadata.get("object_name", "unknown")
    inference_dir = find_inference_dir(run_dir)
    mesh_ready = "yes" if inference_dir and (inference_dir / "object.glb").exists() else "no"
    open_ready = "yes" if (run_dir / "open.png").exists() else "no"
    return (
        f"### Run Ready\n"
        f"- Output dir: `{run_dir}`\n"
        f"- Object: `{object_name}`\n"
        f"- Parts: `{num_parts}`\n"
        f"- 3D mesh ready: `{mesh_ready}`\n"
        f"- Open image ready: `{open_ready}`"
    )


def collect_outputs(run_dir: Path, logs: str = ""):
    metadata = read_json_if_exists(run_dir / "metadata.json")
    prompt_part = read_json_if_exists(run_dir / "prompt_part.json")
    prompt_open = read_json_if_exists(run_dir / "prompt_open.json")
    urdf_text = read_text_if_exists(run_dir / "mobility.urdf")
    render_image = choose_render_image(run_dir)
    open_image = str(run_dir / "open.png") if (run_dir / "open.png").exists() else None
    model_path = choose_model_path(run_dir)
    gallery = list_part_images(run_dir)
    bundle = build_archive(run_dir)
    status = summarize_run(run_dir)
    return (
        status,
        str(run_dir),
        render_image,
        open_image,
        model_path,
        gallery,
        metadata,
        prompt_part,
        prompt_open,
        urdf_text,
        logs,
        bundle,
    )


def generate_part_images(run_dir: Path, env: dict[str, str], logs: list[str]) -> None:
    prompt_part = read_json_if_exists(run_dir / "prompt_part.json")
    prompts = (prompt_part or {}).get("prompts", {})
    if not prompts:
        raise DemoError("No valid prompts were found in `prompt_part.json`.")
    for part_label in prompts:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "VLMguidance" / "generate_image_part.py"),
            "--input",
            str(run_dir),
            "--part",
            part_label,
        ]
        if env.get("OPENAI_API_KEY"):
            command.extend(["--api-key", env["OPENAI_API_KEY"]])
        logs.append(run_command(command, env, f"Generate image for {part_label}"))


def run_open_image_generation(run_dir: Path, env: dict[str, str], openai_model: str, seed: int, logs: list[str]) -> None:
    prompt_command = [
        sys.executable,
        str(PROJECT_ROOT / "VLMguidance" / "generate_prompt_open.py"),
        "--input",
        str(run_dir),
        "--model",
        openai_model,
        "--seed",
        str(seed),
    ]
    if env.get("OPENAI_API_KEY"):
        prompt_command.extend(["--api-key", env["OPENAI_API_KEY"]])
    logs.append(run_command(prompt_command, env, "Generate open prompt"))

    image_command = [
        sys.executable,
        str(PROJECT_ROOT / "VLMguidance" / "generate_image_open.py"),
        "--input",
        str(run_dir),
    ]
    if env.get("OPENAI_API_KEY"):
        image_command.extend(["--api-key", env["OPENAI_API_KEY"]])
    logs.append(run_command(image_command, env, "Generate open image"))


def run_spark_pipeline(
    image_path: str,
    openai_api_key: str,
    openai_model: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    cuda_device: str,
    generate_open_image: bool,
    progress: gr.Progress = gr.Progress(),
):
    if not image_path:
        raise gr.Error("Please upload an image first.")

    try:
        ensure_api_keys(openai_api_key)
        env = prepare_env(openai_api_key, cuda_device)
        run_dir = create_run_dir(image_path)
        copied_image = copy_input_image(image_path, run_dir)
        logs: list[str] = [f"Run directory: {run_dir}", f"Saved input image: {copied_image}"]

        progress(0.05, desc="Preparing input")

        command = [
            sys.executable,
            str(PROJECT_ROOT / "VLMguidance" / "generate_json.py"),
            "--input",
            str(copied_image),
            "--output",
            str(run_dir / "metadata.json"),
            "--csv",
            str(CSV_PATH),
            "--model",
            openai_model,
            "--seed",
            str(seed),
        ]
        if env.get("OPENAI_API_KEY"):
            command.extend(["--api-key", env["OPENAI_API_KEY"]])
        logs.append(run_command(command, env, "Generate metadata"))

        progress(0.18, desc="Fixing joint axes")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "VLMguidance" / "repredict_axis.py"),
            "--input",
            str(run_dir),
            "--image",
            "input.png",
            "--model",
            openai_model,
            "--seed",
            str(seed),
            "--single",
        ]
        if env.get("OPENAI_API_KEY"):
            command.extend(["--api-key", env["OPENAI_API_KEY"]])
        logs.append(run_command(command, env, "Repredict axis"))

        progress(0.28, desc="Generating URDF")
        logs.append(
            run_command(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "VLMguidance" / "generate_urdf.py"),
                    "--input-dir",
                    str(run_dir),
                    "--single",
                ],
                env,
                "Generate URDF",
            )
        )

        progress(0.40, desc="Generating part prompts")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "VLMguidance" / "generate_prompt_part.py"),
            "--input",
            str(run_dir),
            "--model",
            openai_model,
            "--seed",
            str(seed),
        ]
        if env.get("OPENAI_API_KEY"):
            command.extend(["--api-key", env["OPENAI_API_KEY"]])
        logs.append(run_command(command, env, "Generate part prompts"))

        progress(0.58, desc="Generating part images")
        generate_part_images(run_dir, env, logs)

        progress(0.78, desc="Running PartCrafter")
        inference_tag = f"gradio_seed{seed}"
        logs.append(
            run_command(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "inference_partcrafter_part.py"),
                    "--image_folder",
                    str(run_dir),
                    "--output_dir",
                    str(run_dir / "inference"),
                    "--tag",
                    inference_tag,
                    "--seed",
                    str(seed),
                    "--num_inference_steps",
                    str(num_inference_steps),
                    "--guidance_scale",
                    str(guidance_scale),
                    "--render",
                ],
                env,
                "Run PartCrafter inference",
            )
        )

        if generate_open_image:
            progress(0.92, desc="Generating open-state image")
            run_open_image_generation(run_dir, env, openai_model, seed, logs)

        progress(1.0, desc="Done")
        return collect_outputs(run_dir, "\n\n".join(logs))
    except DemoError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def load_existing_result(run_dir_text: str):
    run_dir = Path(run_dir_text).expanduser()
    if not run_dir.exists():
        raise gr.Error(f"Directory does not exist: {run_dir}")
    return collect_outputs(run_dir)


def load_builtin_example():
    if not EXAMPLE_RUN_DIR.exists():
        raise gr.Error(f"Built-in example directory does not exist: {EXAMPLE_RUN_DIR}")
    return collect_outputs(EXAMPLE_RUN_DIR)


with gr.Blocks(
    title="SPARK Demo",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(HEADER)
    gr.HTML(HERO_HTML)

    with gr.Row():
        with gr.Column(scale=5, elem_classes=["panel-card"]):
            gr.HTML(
                """
                <div class="section-title">
                  <h2>Input & Controls</h2>
                  <p>Upload an image, adjust the controls, and launch the full SPARK workflow from here.</p>
                </div>
                """
            )
            input_image = gr.Image(label="Input Image", type="filepath", height=320)

            gr.Examples(
                examples=[[str(EXAMPLE_IMAGE)]] if EXAMPLE_IMAGE.exists() else [],
                inputs=[input_image],
                cache_examples=False,
                label="Examples",
            )

            with gr.Row():
                run_button = gr.Button("Generate 3D Model", variant="primary", size="lg")
                example_button = gr.Button("▶ Run Example", variant="secondary", size="lg")

            gr.HTML('<div class="subsection-title">3D Generation Controls</div>')
            with gr.Row():
                seed = gr.Slider(label="Generation Seed", minimum=0, maximum=9999, step=1, value=DEFAULT_SEED)
                guidance_scale = gr.Slider(label="CFG Strength", minimum=0.0, maximum=10.0, step=0.5, value=DEFAULT_GUIDANCE_SCALE)

            with gr.Row():
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=50, maximum=1500, step=50, value=DEFAULT_INFERENCE_STEPS)

            with gr.Accordion("Model & API Settings", open=False):
                gr.Markdown(
                    "**An `OpenAI API Key` is required for the full SPARK pipeline.**  \n"
                    "OpenAI handles metadata and prompt generation as well as part/open-state image generation (gpt-image-2).",
                    elem_classes=["hint-text"],
                )
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...", value="")
                openai_model = gr.Textbox(label="OpenAI Model", value=DEFAULT_OPENAI_MODEL)
                cuda_device = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value=DEFAULT_CUDA_DEVICE)
                generate_open_image = gr.Checkbox(label="Also generate open-state image", value=True)

            with gr.Accordion("Workflow Notes", open=False):
                gr.Markdown(
                    """
                    - Keep the part count modest, otherwise part-image generation and 3D inference will take much longer.
                    - The right-side result area is designed to feel closer to OmniPart's display layout.
                    - If you already have previous outputs, load the directory directly instead of calling the APIs again.
                    """
                )

            with gr.Accordion("Load Existing Output Directory", open=False):
                existing_run_dir = gr.Textbox(label="Existing Output Directory", placeholder=str(EXAMPLE_RUN_DIR))
                load_existing_button = gr.Button("Load Output Directory")

        with gr.Column(scale=7, elem_classes=["panel-card"]):
            gr.HTML(
                """
                <div class="section-title">
                                    <h2>Results Panel</h2>
                                    <p>Preview 2D outputs, inspect 3D assets, and review structured files in a single workspace.</p>
                </div>
                """
            )
            status_markdown = gr.Markdown(value="### Ready\n- Load the built-in example or upload an image to begin.", elem_classes=["status-card"])

            with gr.Row(elem_classes=["results-top-row"]):
                render_image = gr.Image(label="Preview Image", height=280)
                open_image = gr.Image(label="Open View", height=280)

            with gr.Row():
                object_model = gr.Model3D(label="3D Model Viewer", height=380)
                part_gallery = gr.Gallery(label="Part Gallery", columns=2, height=380, object_fit="contain")

            with gr.Tabs(elem_classes=["compact-tabs"]):
                with gr.Tab("Data View"):
                    with gr.Row():
                        metadata_json = gr.JSON(label="metadata.json")
                        prompt_part_json = gr.JSON(label="prompt_part.json")
                    with gr.Row():
                        prompt_open_json = gr.JSON(label="prompt_open.json")
                        urdf_code = gr.Code(label="mobility.urdf", language="html")

                with gr.Tab("Run Logs"):
                    run_dir_box = gr.Textbox(label="Output Directory")
                    bundle_file = gr.File(label="Download Result Zip")
                    logs_box = gr.Textbox(label="Execution Logs", lines=18)

    with gr.Row(elem_classes=["examples-panel"]):
        with gr.Column(elem_classes=["examples-strip-card"]):
            gr.HTML(
                """
                <div class="examples-strip-title">
                                    <h3>Example Strip</h3>
                                    <p>Pick an example below, then click <strong>▶ Run Example</strong> to load the reference demo result.</p>
                </div>
                """
            )
            gr.Examples(
                examples=[[str(EXAMPLE_IMAGE)]] if EXAMPLE_IMAGE.exists() else [],
                inputs=[input_image],
                cache_examples=False,
                                label="Examples",
            )

    outputs = [
        status_markdown,
        run_dir_box,
        render_image,
        open_image,
        object_model,
        part_gallery,
        metadata_json,
        prompt_part_json,
        prompt_open_json,
        urdf_code,
        logs_box,
        bundle_file,
    ]

    run_button.click(
        fn=run_spark_pipeline,
        inputs=[
            input_image,
            openai_api_key,
            openai_model,
            seed,
            num_inference_steps,
            guidance_scale,
            cuda_device,
            generate_open_image,
        ],
        outputs=outputs,
    )

    example_button.click(fn=load_builtin_example, outputs=outputs)
    load_existing_button.click(fn=load_existing_result, inputs=[existing_run_dir], outputs=outputs)


def main() -> None:
    demo.queue(max_size=8)
    demo.launch(server_name="0.0.0.0", share=True, show_error=True)


if __name__ == "__main__":
    main()
