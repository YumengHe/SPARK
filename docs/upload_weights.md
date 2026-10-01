# Pushing Pretrained Weights to HuggingFace

For maintainers. After (re-)training, upload the contents of `pretrained_weights/` to [`2bidoubi/SPARK`](https://huggingface.co/2bidoubi/SPARK) so that users auto-download them via `src.utils.download_weights`.

**1. Get a write token** at https://huggingface.co/settings/tokens (Type → **Write**).

**2. Log in.**

```bash
hf auth login
```

**3. Install the Xet backend** (5–10× faster uploads for large files):

```bash
pip install hf_xet
```

**4. Upload.** The script auto-creates the repo and uses `upload_large_folder` (parallel, resumable, auto-LFS):

```bash
python scripts/tools/upload_weights.py
```

If the upload is interrupted, just rerun the same command — already-uploaded files are skipped via SHA comparison.

**5. Verify.** Browse to https://huggingface.co/2bidoubi/SPARK and confirm `PartCrafter/`, `TripoSG/`, `RMBG-1.4/` all appear.
