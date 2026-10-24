# GPT-2 RLHF demo

Set up the Python 3.12 environment with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python gpt2_rlhf.py
```

The first run downloads GPT-2 from Hugging Face, then trains the models.
PyTorch uses CUDA when available and otherwise runs on the CPU.
Model outputs are saved in `gpt2_sft/` and `gpt2_rlhf_result/`.

Compare the original GPT-2 and the trained model interactively:

```bash
python compare.py
```

Or compare a single question:

```bash
python compare.py "What is 2 + 2?"
python compare.py --after gpt2_sft "What is the capital of France?"
```

Each question uses the same `Question: ... Answer:` format as training and
deterministic generation for both models. Type `/quit` to exit interactive mode.
Use `--max-new-tokens 100` for longer answers or `--device cpu` to run on CPU.
