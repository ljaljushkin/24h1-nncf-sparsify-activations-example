# 24h1-nncf-sparsify-activations-example

## Setup

Python 3.9+ is required.

```bash
git clone https://github.com/nikita-savelyevv/nncf.git /tmp/nncf # install not inside repository to avoid import issues
cd /tmp/nncf && git checkout activation-sparsity-ov-backend
pip install -e .
cd -
pip install -r requirements.txt
pip install pip==22.0 # to avoid issue with deprecation of --build-option in pip==23.0
```

## Run
Example command:
```bash
python run_sparsify_activations.py \
--model_id meta-llama/Llama-2-7b-hf \
--torch_dtype float32 --backend ov --device cpu \
--compress_weights_mode int8_asym \
--up 0.32 --gate 0.32 --down 0.52 \
--save_folder ./models/llama2-7b_int8-asym_sparse
```
