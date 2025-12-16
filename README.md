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

## Building openvino.genai with custom OpenVINO

To run llm_bench with a specific OpenVINO build (e.g., commit `19b53c43b67` from January 3, 2025), you need to build openvino.genai from sources.

### Prerequisites
1. Build OpenVINO from source
2. Clone openvino.genai and checkout matching commit (e.g., `42f3053afdaa` from Jan 3, 2025)

### Build Steps

1. **Build openvino.genai** with OpenVINO_DIR pointing to your OpenVINO build:
   ```bash
   cd /home/nlyaly/projects/openvino.genai
   git checkout 42f3053afdaa  # Use commit matching your OpenVINO build date
   git submodule update --init --recursive

   OpenVINO_DIR=/home/nlyaly/projects/openvino/build pip install .
   ```

2. **Build openvino_tokenizers** from the submodule (required for compatibility):
   ```bash
   cd /home/nlyaly/projects/openvino.genai/thirdparty/openvino_tokenizers
   pip uninstall -y openvino-tokenizers
   OpenVINO_DIR=/home/nlyaly/projects/openvino/build pip install .
   ```

3. **Set LD_LIBRARY_PATH at runtime** to use the OpenVINO libraries from your build:
   ```bash
   export LD_LIBRARY_PATH=/home/nlyaly/projects/openvino/bin/intel64/Release:$LD_LIBRARY_PATH
   cd /home/nlyaly/projects/openvino.genai/tools/llm_bench
   python benchmark.py -h
   ```

4. **Or use the wrapper script**:
   ```bash
   cd /home/nlyaly/projects/openvino.genai/tools/llm_bench
   ./run_benchmark_with_custom_ov.sh -h
   ```

### Key Points
- Both `openvino-genai` and `openvino-tokenizers` must be built against the same custom OpenVINO
- openvino-genai needs to find the OpenVINO shared libraries at runtime
- When built from sources, those libraries are in `/home/nlyaly/projects/openvino/bin/intel64/Release`
- The `OpenVINO_DIR` environment variable tells CMake where to find OpenVINO during build
- The `LD_LIBRARY_PATH` environment variable tells the runtime where to find OpenVINO shared libraries

