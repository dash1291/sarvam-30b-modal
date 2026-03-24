FROM vllm/vllm-openai:latest

# Install newer transformers for MoE models
RUN pip install --no-cache-dir --upgrade "transformers>=5.0.0"

