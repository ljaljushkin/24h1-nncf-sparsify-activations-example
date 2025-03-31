import json
import csv
import re
from pathlib import Path

from tqdm import tqdm

root_dir = Path("/media/hdd1/models/sparsity_08032025/")


results = []
for model_file in sorted(root_dir.rglob("openvino_model.bin")):
    if "tiny-random-LlamaForCausalLM" in str(model_file):
        continue

    model_dir = model_file.parent

    benchmark_json_file = model_dir / "llm_bench_report_n3.json"
    lm_eval_json_file = model_dir / "ov_eval_results.json"
    if not benchmark_json_file.exists() or not lm_eval_json_file.exists():
        print("Skipping", model_dir)
        continue

    with open(benchmark_json_file, "r") as f:
        data = json.load(f)
        first_latency = data["perfdata"]["results_averaged"]["first_latency"]
        second_avg_latency = data["perfdata"]["results_averaged"]["second_avg_latency"]

    with open(lm_eval_json_file, "r") as f:
        data = json.load(f)
        acc = data["results"]["mmlu"]["acc,none"]

    model_path = str(model_dir).replace(str(root_dir), "")
    dash_split = model_path.split("/")[1:]
    model_id = dash_split[0]
    is_sparse = False
    up_ratio = gate_ratio = down_ratio = None
    if "sparse" in model_path:
        is_sparse = True
        sparsity_str = dash_split[1]
        match = re.search(r'up-([0-9.]+)-gate-([0-9.]+)-down-([0-9.]+)', sparsity_str)
        up_ratio = float(match.group(1))
        gate_ratio = float(match.group(2))
        down_ratio = float(match.group(3))
    ratio = 1.0
    if "ratio" in model_path:
        pattern = r'ratio-([0-9.]+)'

        match = re.search(pattern, model_path)
        ratio = float(match.group(1))
    group_size = 128
    if "_gs-" in model_path:
        pattern = r'_gs-([0-9]+)'
        match = re.search(pattern, model_path)
        group_size = int(match.group(1))
    is_int4 = "int4-default" in dash_split[-1]
    is_int8 = "int8-asym" in dash_split[-1]
    is_bf16 = not(is_int4 or is_int8)

    # Format float numbers to 4 digits and add row
    model_size = model_file.stat().st_size / 2**30  # In billions of parameters
    results.append(
        [
            model_id,
            f"{model_size:.2f}".replace(".", ","),
            "bf16" if is_bf16 else "int8" if is_int8 else "int4",
            f"{ratio:.2f}".replace(".", ","),
            f"{group_size}",
            is_sparse,
            f"{up_ratio:.4f}".replace(".", ",") if up_ratio is not None else "",
            f"{gate_ratio:.4f}".replace(".", ",") if gate_ratio is not None else "",
            f"{down_ratio:.4f}".replace(".", ",") if down_ratio is not None else "",
            f"{first_latency:.2f}".replace(".", ","),
            f"{second_avg_latency:.2f}".replace(".", ","),
            f"{acc:.4f}".replace(".", ","),
        ]
    )

column_names = ["model_id", "model_size", "weights", "int4_ratio", "group_size", "is_sparse", "up_ratio", "gate_ratio",
                "down_ratio", "first_latency", "second_avg_latency", "acc"]
with open("dynamic_sparsity.csv", "w") as f:
    # use csv package to dump table
    writer = csv.writer(f)
    writer.writerow(column_names)
    writer.writerows(results)
