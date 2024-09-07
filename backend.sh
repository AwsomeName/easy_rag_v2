
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model /home/lc/cv_models/Qwen2-0.5B \
    --trust-remote-code \
    --dtype auto \
    --served-model-name qwen \
    --port 3332 \
    --tensor-parallel-size=1 \
    2>&1 1>vllm-3332.log &