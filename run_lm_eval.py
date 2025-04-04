import subprocess
from pathlib import Path

from tqdm import tqdm

lm_eval_path = "/home/nsavel/venvs/llm_bench_dynsparse_20250/bin/lm-eval"
root_dir = Path("/media/hdd1/models/sparsity_08032025")

# Filter out paths:
model_dir_paths = []
for model_file in reversed(sorted(root_dir.rglob("openvino_model.xml"))):
    model_dir = model_file.parent
    if "tiny-random-LlamaForCausalLM" in str(model_dir):
        continue
    # if "up-0.20-gate-0.20-down-0.30" not in str(model_dir):
    #     continue
    if (model_dir / "mmlu_eval").exists():
        continue
    model_dir_paths.append(model_dir)
for model_dir in tqdm(model_dir_paths):
    run_command = f"{lm_eval_path} --model openvino --model_args pretrained={model_dir},trust_remote_code=True --tasks mmlu --output_path {model_dir}/mmlu_eval"
    process = subprocess.Popen(run_command, shell=True)
    process.wait()
