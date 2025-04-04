#!/bin/bash

# Define parallel arrays for model_id1 and model_id2
model_ids_1=("HuggingFaceH4" "microsoft" "Qwen" "meta-llama" "deepseek-ai")
model_ids_2=("tiny-random-LlamaForCausalLM" "Phi-4-mini-instruct" "Qwen2.5-7B-Instruct" "Llama-3.1-8B-Instruct" "DeepSeek-R1-Distill-Llama-8B")
ups=("0.20" "0.30" "0.40")
downs=("0.30" "0.40" "0.50")

#model_ids_1=("microsoft" "Qwen" "meta-llama")
#model_ids_2=("Phi-4-mini-instruct" "Qwen2.5-7B-Instruct" "Llama-3.1-8B-Instruct")
#additional_args=("" "" "" "")
#ratios=("0.75" "0.5" "0.25")

#model_ids_1=("Qwen")
#model_ids_2=("Qwen2.5-32B-Instruct")
#additional_args=("")
#ups=("0.20")
#downs=("0.30")
#group_sizes=("32" "128" "256")

# Set constant parameters
task="mmlu"
save_dir="/media/hdd1/models/sparsity_08032025"

# Check that both arrays have the same length
if [ "${#model_ids_1[@]}" -ne "${#model_ids_2[@]}" ]; then
  echo "Error: Arrays model_ids_1 and model_ids_2 must have the same number of elements."
  exit 1
fi

# Loop over the indices of the arrays
for i in "${!model_ids_1[@]}"; do
  m1="${model_ids_1[i]}"
  m2="${model_ids_2[i]}"
  args="${additional_args[i]}"
  model_id="$m1/$m2"
  models_dir="$save_dir/$m2"
  
  echo "Processing model: $model_id"
  python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
    --save_folder "$models_dir/bf16"
  python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
    --compress_weights_mode int8_asym --save_folder "$models_dir/int8-asym"
  python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
    --compress_weights_mode int4 --save_folder "$models_dir/int4-default"

#  for j in "${!ratios[@]}"; do
#    ratio="${ratios[j]}"
#    python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
#      --compress_weights_mode int4 --ratio "$ratio" --save_folder "$models_dir/int4-default_ratio-$ratio"
#  done
#  for j in "${!group_sizes[@]}"; do
#    group_size="${group_sizes[j]}"
#    python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
#      --compress_weights_mode int4 --group_size "$group_size" --save_folder "$models_dir/int4-default_gs-$group_size"
#  done

  for j in "${!ups[@]}"; do
    up="${ups[j]}"
    gate="${ups[j]}"
    for k in "${!downs[@]}"; do
      down="${downs[k]}"
      models_dir="$save_dir/$m2/up-$up-gate-$gate-down-$down"

      echo "Processing model: $model_id, up: $up, gate: $gate, down: $down"
      python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
        --up "$up" --gate "$gate" --down "$down" --save_folder "$models_dir/bf16_sparse"
      python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
        --compress_weights_mode int8_asym --up "$up" --gate "$gate" --down "$down" --save_folder "$models_dir/int8-asym_sparse"
      python run_sparsify_activations.py --model-id "$model_id" --backend ov --device cpu --eval-task "$task" $args \
        --compress_weights_mode int4 --up "$up" --gate "$gate" --down "$down" --save_folder "$models_dir/int4-default_sparse"
    done
  done
done