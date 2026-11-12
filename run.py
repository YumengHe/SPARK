import subprocess
import os
import json
import glob
import shutil
import time
from PIL import Image

# Run all images in INPUT_DIR or specific named images
RUNALL = True  # True = process all images in INPUT_DIR, False = process only NAMES

# List of image names to process (stem without extension, used when RUNALL = False)
NAMES = ["10"]

# Variables - modify these as needed
INPUT_DIR = "./image"  # directory containing input images
OUTPUT_DIR = "./output"    # directory where all outputs are stored
CSV_PATH = "./VLMguidance/partnet-mobility-data-analysis.csv"
SEED = 42

# Inference parameters (Step 6)
CUDA_DEVICE = "0"
INFERENCE_SEED = 0
GUIDANCE_SCALE = 1
INFERENCE_TAG = f"FT5_G1_{INFERENCE_SEED}"
NUM_INFERENCE_STEPS = 1000

# Steps
step1 = 1 # STEP 1: Generate metadata.json
step2 = 1 # STEP 2: Fix metadata.json
step3 = 1 # STEP 3: Generate URDF
step4 = 1 # STEP 4: Generate part image prompts
step5 = 1 # STEP 5: Generate images for each part
step6 = 1 # STEP 6: Run inference
step7 = 1 # STEP 7: Clean mesh
step8 = 1 # STEP 8: Generate open image prompts
step8b = 1 # STEP 8b: Generate open image (open.png)
step9 = 1 # STEP 9: Generate texture

# Create OUTPUT_DIR if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Timestamp shared by every image processed in this run so re-running keeps prior results
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpeg', '.jpg')

# Determine which images to process
if RUNALL:
    names_to_process = []
    for fname in sorted(os.listdir(INPUT_DIR)):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            names_to_process.append(os.path.splitext(fname)[0])
    print(f"RUNALL mode: Found {len(names_to_process)} images")
else:
    names_to_process = NAMES

# Process each image
for name in names_to_process:
    # Find the input image file
    input_image = None
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(INPUT_DIR, f"{name}{ext}")
        if os.path.exists(candidate):
            input_image = candidate
            break

    if input_image is None:
        print(f"Warning: No image found for '{name}' in {INPUT_DIR}, skipping...")
        continue

    # Create output subfolder named after the image + run timestamp
    output_subdir = os.path.join(OUTPUT_DIR, f"{name}_{RUN_TIMESTAMP}")
    os.makedirs(output_subdir, exist_ok=True)

    # Copy/convert input image to output subfolder as input.png
    input_path = os.path.join(output_subdir, "input.png")
    if input_image.lower().endswith('.png'):
        shutil.copy2(input_image, input_path)
    else:
        img = Image.open(input_image)
        img.save(input_path)

    # STEP 1: Generate metadata.json
    if step1:
        cmd = [
            "python", "VLMguidance/generate_json.py",
            "--input", input_path,
            "--output", os.path.join(output_subdir, "metadata.json"),
            "--csv", CSV_PATH,
            "--seed", str(SEED)
        ]
        print(f"Processing: {name}")
        subprocess.run(cmd)
    else:
        print(f"Skip Step 1")

    # STEP 2: Fix metadata.json
    if step2:
        cmd2 = [
            "python", "VLMguidance/repredict_axis.py",
            "--input", output_subdir,
            "--image", "input.png",
            "--single"
        ]
        print(f"Fixing axis for: {name}")
        subprocess.run(cmd2)
    else:
        print(f"Skip Step 2")

    # STEP 3: Generate URDF
    if step3:
        cmd3 = [
            "python", "VLMguidance/generate_urdf.py",
            "--input-dir", output_subdir,
            "--single"
        ]
        print(f"Generating URDF for: {name}")
        subprocess.run(cmd3)
    else:
        print(f"Skip Step 3")

    # STEP 4: Generate part image prompts
    if step4:
        cmd4 = [
            "python", "VLMguidance/generate_prompt_part.py",
            "--input", output_subdir
        ]
        print(f"Generating part prompt for: {name}")
        subprocess.run(cmd4)
    else:
        print(f"Skip Step 4")

    # STEP 5: Generate images for each part
    if step5:
        prompt_part_path = os.path.join(output_subdir, "prompt_part.json")
        if os.path.exists(prompt_part_path):
            with open(prompt_part_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)

            parts = prompt_data.get("prompts", {}).keys()

            for part_label in parts:
                cmd5 = [
                    "python", "VLMguidance/generate_image_part.py",
                    "--input", output_subdir,
                    "--part", part_label
                ]

                subprocess.run(cmd5)
        else:
            print(f"Warning: prompt_part.json not found for '{name}', skipping image generation...")
    else:
        print(f"Skip Step 5")

    # STEP 6: Run inference
    if step6:
        inference_output = os.path.join(output_subdir, "inference")

        cmd6 = [
            "python", "scripts/inference/part_images.py",
            "--image_folder", output_subdir,
            "--output_dir", inference_output,
            "--tag", INFERENCE_TAG,
            "--seed", str(INFERENCE_SEED),
            "--num_inference_steps", str(NUM_INFERENCE_STEPS),
            "--guidance_scale", str(GUIDANCE_SCALE),
            "--render"
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
        # Add project root to PYTHONPATH so 'src' module can be found
        project_root = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        print(f"Running inference for: {name}")
        subprocess.run(cmd6, env=env)
    else:
        print(f"Skip Step 6")

    # STEP 7: Clean mesh
    if step7:
        inference_tag_dir = os.path.join(output_subdir, "inference", INFERENCE_TAG)
        mesh_files = glob.glob(os.path.join(inference_tag_dir, "part_*.glb"))

        for mesh_file in mesh_files:
            base_name = os.path.splitext(mesh_file)[0]
            ext = os.path.splitext(mesh_file)[1]
            cleaned_mesh = f"{base_name}_cleaned{ext}"

            cmd7 = [
                "python", "src/utils/clean_mesh.py",
                "--in_mesh", mesh_file,
                "--out_mesh", cleaned_mesh,
                "--keep_largest",
                "--merge_tol", "1e-5",
                "--target_faces", "200000",
            ]

            print(f"Cleaning mesh: {mesh_file}")
            subprocess.run(cmd7)
    else:
        print(f"Skip Step 7")

    # STEP 8: Generate open image prompts
    if step8:
        cmd8 = [
            "python", "VLMguidance/generate_prompt_open.py",
            "--input", output_subdir
        ]
        print(f"Generating open prompt for: {name}")
        subprocess.run(cmd8)
    else:
        print(f"Skip Step 8")

    # STEP 8b: Generate open image (open.png)
    if step8b:
        cmd8b = [
            "python", "VLMguidance/generate_image_open.py",
            "--input", output_subdir
        ]
        print(f"Generating open image for: {name}")
        subprocess.run(cmd8b)
    else:
        print(f"Skip Step 8b")

    # STEP 9: Generate texture
    if step9:
        inference_tag_dir = os.path.join(output_subdir, "inference", INFERENCE_TAG)
        mesh_files = glob.glob(os.path.join(inference_tag_dir, "part_*_cleaned.glb"))

        for mesh_file in mesh_files:
            part_name = os.path.splitext(os.path.basename(mesh_file))[0]
            texture_outdir = os.path.join(inference_tag_dir, f"{part_name}_texture")

            # Step 9a: Generate texture via Meshy
            cmd9a = [
                "python", "VLMguidance/generate_texture.py",
                "--image", input_path,
                "--model", mesh_file,
                "--outdir", texture_outdir,
            ]
            print(f"Generating texture for: {mesh_file}")
            subprocess.run(cmd9a)

            # Step 9b: Align textured GLB back to original cleaned mesh
            textured_glbs = glob.glob(os.path.join(texture_outdir, "**/*.glb"), recursive=True)
            if textured_glbs:
                textured_glb = textured_glbs[0]
                aligned_glb = os.path.join(inference_tag_dir, f"{part_name}_textured_aligned.glb")
                cmd9b = [
                    "python", "VLMguidance/align_glb.py",
                    "--reference", mesh_file,
                    "--target", textured_glb,
                    "--output", aligned_glb,
                ]
                print(f"Aligning textured GLB: {textured_glb}")
                subprocess.run(cmd9b)
            else:
                print(f"Warning: no textured GLB found in {texture_outdir}, skipping alignment...")
    else:
        print(f"Skip Step 9")
