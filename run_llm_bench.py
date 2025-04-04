import subprocess
from pathlib import Path

from tqdm import tqdm

python_path = "/home/nsavel/venvs/llm_bench_dynsparse_20250/bin/python"
benchmark_py_path = "/home/nsavel/workspace/openvino.genai/tools/llm_bench/benchmark.py"
root_dir = Path("/media/hdd1/models/sparsity_08032025")
report_file_name = "llm_bench_report_n3.json"


for model_file in tqdm(sorted(root_dir.rglob("openvino_model.xml"))):
    model_dir = model_file.parent
    if (model_dir / report_file_name).exists() or "tiny-random-LlamaForCausalLM" in str(model_dir):
        continue
    run_command = f"{python_path} {benchmark_py_path} -m {model_dir} -rj {model_dir}/{report_file_name} -n 3 -ic 128"
    process = subprocess.Popen(run_command, shell=True)
    process.wait()
