"""Compare original GPT-2 with the saved trained model on your own prompts."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Load both checkpoints once, then compare one question or an input loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Question to compare; omit for interactive mode")
    parser.add_argument("--after", type=Path,
                        default=Path(__file__).resolve().parent / "gpt2_rlhf_result",
                        help="Saved model directory (default: gpt2_rlhf_result)")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not (args.after / "config.json").is_file():
        parser.error(f"No trained model in {args.after}. Run gpt2_rlhf.py first.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    print(f"Loading models on {device}...")
    models = []
    for label, source in [("Before", "openai-community/gpt2"), ("After", str(args.after))]:
        # Use each checkpoint's tokenizer so its vocabulary matches its weights.
        # GPT-2 has no dedicated padding token; explicit tokenizer masks below
        # distinguish actual input tokens from any padding, even when PAD = EOS.
        tokenizer = AutoTokenizer.from_pretrained(source)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(source).to(device).eval()
        models.append((label, tokenizer, model))

    def compare(question):
        # Match the training template; arbitrary chat formatting was not trained.
        text = f"Question: {question}\nAnswer:"
        # Validate both context windows before generating either answer.
        inputs = []
        for label, tokenizer, model in models:
            encoded = tokenizer(text, return_tensors="pt").to(device)
            length = encoded.input_ids.shape[1]
            if length + args.max_new_tokens > model.config.max_position_embeddings:
                print(f"Prompt too long for {label}; shorten it or reduce --max-new-tokens.")
                return
            inputs.append(encoded)
        for (label, tokenizer, model), encoded in zip(models, inputs):
            # No gradients are needed for inference. Greedy decoding removes
            # sampling randomness so differences reflect the model checkpoints.
            # **encoded includes attention_mask as well as input_ids.
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            # generate() returns the prompt followed by its continuation; show
            # only newly generated tokens, omitting EOS and other special tokens.
            answer = tokenizer.decode(
                output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            print(f"\n{label}: {answer or '(empty response)'}")
        print()

    if args.prompt is not None:
        compare(args.prompt)
        return
    print("Enter a question to compare. Type /quit to exit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/quit":
            break
        if question:
            compare(question)


if __name__ == "__main__":
    main()
