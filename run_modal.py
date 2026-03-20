"""
Launch parameter-golf training on a single Modal T4 GPU.

Usage:
    modal run run_modal.py

Override hyperparameters via environment variables:
    ITERATIONS=500 modal run run_modal.py
"""

import os

import modal

app = modal.App("parameter-golf")

# Build the image with all dependencies and the dataset baked in so we don't
# re-download on every run.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "tqdm",
        "torch",
        "huggingface-hub",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
        "datasets",
        "tiktoken",
        "sentencepiece",
    )
    # Copy the full repo into the image so train_gpt.py and data scripts are available.
    .add_local_dir(".", remote_path="/workspace/parameter-golf", ignore=[".venv", ".git", "logs", "*.pt", "*.ptz"])
    # Download the dataset at build time so it's cached in the image layer.
    .run_commands(
        "cd /workspace/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
    )
)

# Volume to persist outputs (model checkpoints, logs) across runs.
output_vol = modal.Volume.from_name("parameter-golf-outputs", create_if_missing=True)

# Environment variables forwarded from the caller (with sensible defaults for a
# single-T4 smoke run).
ENV_DEFAULTS = {
    "RUN_ID": "modal_t4_run",
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "TRAIN_BATCH_TOKENS": "131072",  # smaller batch for T4's 16 GB VRAM
    "VAL_BATCH_SIZE": "131072",
    "VAL_LOSS_EVERY": "500",
    "TRAIN_LOG_EVERY": "100",
    "MAX_WALLCLOCK_SECONDS": "600",
}

# Let the caller override any hyperparameter via local env vars.
FORWARDED_ENVS = [
    "RUN_ID", "SEED", "ITERATIONS", "WARMDOWN_ITERS", "WARMUP_STEPS",
    "TRAIN_BATCH_TOKENS", "TRAIN_SEQ_LEN", "MAX_WALLCLOCK_SECONDS",
    "VOCAB_SIZE", "NUM_LAYERS", "NUM_KV_HEADS", "MODEL_DIM", "NUM_HEADS",
    "MLP_MULT", "TIE_EMBEDDINGS", "ROPE_BASE", "LOGIT_SOFTCAP",
    "EMBED_LR", "HEAD_LR", "TIED_EMBED_LR", "TIED_EMBED_INIT_STD",
    "MATRIX_LR", "SCALAR_LR", "MUON_MOMENTUM", "MUON_BACKEND_STEPS",
    "VAL_BATCH_SIZE", "VAL_LOSS_EVERY", "TRAIN_LOG_EVERY",
    "QK_GAIN_INIT", "GRAD_CLIP_NORM",
]


def _build_env() -> dict[str, str]:
    env = dict(ENV_DEFAULTS)
    for key in FORWARDED_ENVS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    return env


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/outputs": output_vol},
)
def train():
    import subprocess

    env = {**os.environ, **_build_env()}

    result = subprocess.run(
        [
            "torchrun", "--standalone", "--nproc_per_node=1",
            "train_gpt.py",
        ],
        cwd="/workspace/parameter-golf",
        env=env,
    )

    # Copy artifacts to the persistent volume.
    import shutil
    for name in ("final_model.pt", "final_model.int8.ptz"):
        src = f"/workspace/parameter-golf/{name}"
        if os.path.exists(src):
            shutil.copy2(src, f"/outputs/{name}")

    output_vol.commit()

    if result.returncode != 0:
        raise SystemExit(f"Training failed with exit code {result.returncode}")


@app.local_entrypoint()
def main():
    train.remote()
