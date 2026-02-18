docker run --rm -it \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v /shared_workspace_mfs/original_models/:/shared_workspace_mfs/original_models/ \
  vllm/vllm-openai:latest \
    /shared_workspace_mfs/original_models/Qwen3-Coder-30B-A3B-Instruct \
    --max-model-len 32000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 8000
