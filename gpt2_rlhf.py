"""Small GPT-2 RLHF demo: supervised training -> reward model -> PPO.

Install: uv pip install --python .venv/bin/python -r requirements.txt
Run:     .venv/bin/python gpt2_rlhf.py

The sample preferences are hand-written stand-ins for human feedback.
This is for seeing how the pieces work, not for building a useful assistant.
"""

import copy
import gc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

random.seed(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
name = "openai-community/gpt2"  # GPT-2 small, 124M parameters

# Each question has a preferred answer and an intentionally worse answer.
examples = [
    ("What is 2 + 2?", "4.", "5."),
    ("What is the capital of France?", "Paris.", "London."),
    ("What planet do humans live on?", "Earth.", "Mars."),
    ("What gas do plants absorb?", "Carbon dioxide.", "Helium."),
]

tok = AutoTokenizer.from_pretrained(name)
tok.pad_token = tok.eos_token


def prompt(question):
    return f"Question: {question}\nAnswer:"


def tokens(text):
    return tok(text, return_tensors="pt").input_ids.to(device)


def attention_mask(input_ids):
    # Every sequence is unpadded (batch size 1). Keep real EOS tokens visible,
    # even though GPT-2 uses the same ID for EOS and padding.
    return torch.ones_like(input_ids)


def answer(model, question):
    model.eval()
    x = tokens(prompt(question))
    with torch.no_grad():
        y = model.generate(x, attention_mask=attention_mask(x), max_new_tokens=16, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(y[0, x.shape[1]:], skip_special_tokens=True).strip()


# 1. SUPERVISED FINE-TUNING: teach GPT-2 what an answer looks like.
# GPT-2 already learned to continue text during its original training. Here we
# show it a few correct answers so it learns our Question/Answer format.
# "Supervised" means we supply the desired answer, rather than asking the model
# to discover it. Only `good` answers are used in this step; preferences come later.
# Four examples can teach memorized responses, not general knowledge or reasoning.

# "Policy" is the model that will eventually generate our answers. Load its
# existing learned weights, then place them on the GPU (or CPU fallback).
# A causal language model predicts the next token using only preceding tokens.
# Tokens are vocabulary IDs representing pieces of text, not necessarily words.
policy = AutoModelForCausalLM.from_pretrained(name).to(device)
# GPT-2 has no separate padding token. Reuse EOS (end of sequence); our explicit
# attention masks ensure actual EOS tokens are still treated as real input.
policy.config.pad_token_id = tok.eos_token_id
# Transformers 4.57.1 does not infer GPT-2's loss name correctly. Select the
# standard next-token loss explicitly, and save that intent in the config.
policy.config.loss_type = "ForCausalLM"
policy.loss_type = "ForCausalLM"
print("Before SFT:", answer(policy, examples[0][0]))
# answer() switches to evaluation mode. Switch back to training mode to enable
# training behavior such as dropout, which randomly hides some activations.
policy.train()
# parameters() are the trainable numbers inside the network. AdamW updates them
# using gradients. lr=5e-5 means a learning rate of 0.00005: small update steps.
optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5)

# One epoch is one pass over all four examples. Twenty epochs = 80 updates here.
for epoch in range(20):
    # Change the order each pass so training does not always follow one sequence.
    random.shuffle(examples)
    for question, good, _ in examples:
        # Each example contains (question, good answer, bad answer). `_` ignores
        # the bad answer because this step trains only on correct continuations.
        # start contains just the question and "Answer:"; full adds the answer
        # and EOS, which teaches the model when it should stop generating.
        start = tokens(prompt(question))
        full = tokens(prompt(question) + " " + good + tok.eos_token)

        # Token tensors have shape [batch size, number of tokens]. Here the batch
        # size is 1. Clone the IDs so changing labels cannot change the input.
        # Labels say which next tokens the model should learn to predict.
        labels = full.clone()
        # `:` selects every batch row; :start.shape[1] selects the prompt tokens.
        # -100 is the loss function's "ignore this target" marker. The model can
        # still READ the whole question, but it is graded only on answer + EOS.
        # An attention mask controls what it can read; labels control its grade.
        labels[:, :start.shape[1]] = -100

        # During training the correct answer is included in the input. At each
        # position, GPT-2 can see only earlier tokens, so it must predict the next
        # one. Transformers shifts predictions and labels internally; do not
        # shift labels ourselves. This is called "teacher forcing": each next
        # prediction uses the correct prefix rather than its own previous guess.
        # Loss is one number measuring how unlikely the correct tokens were.
        # Lower loss means the model assigns them higher probability.
        loss = policy(full, attention_mask=attention_mask(full), labels=labels).loss
        # Gradients accumulate by default. Clear the previous example's values.
        optimizer.zero_grad()
        # Backpropagation computes how each parameter affects this loss. This
        # computes gradients; it does not change the model's weights yet.
        loss.backward()
        # Apply an update using those gradients to make this answer more likely.
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        # item() turns a one-number tensor into a Python number for printing.
        # This is the LAST example's loss, not an average over the whole epoch.
        print(f"SFT epoch {epoch + 1}: loss={loss.item():.3f}")

# Save weights/config and tokenizer together so the model can be loaded later
# by compare.py or used as the starting point for another training stage.
policy.save_pretrained("gpt2_sft")
tok.save_pretrained("gpt2_sft")
# AdamW stores extra tensors for its updates. We are done with this optimizer;
# release it before creating the reward model to reduce memory use.
del optimizer
gc.collect()
if device == "cuda":
    # Release unused cached GPU allocations, not memory used by live models.
    torch.cuda.empty_cache()
# Generate an answer without supplying the correct answer, unlike training.
# examples was shuffled, so this may be a different question than "Before SFT".
print("After SFT:", answer(policy, examples[0][0]))


# 2. REWARD MODEL: score the good answer above the bad one.
# Step 1 learned to produce our example answers. Now a separate model learns
# to judge answers: given a question AND an answer, output one numeric score.
# In real RLHF, people supply these preferences. Here we wrote the pairs by hand.
# A score is not a probability or a guarantee of correctness; its ordering is
# what matters. We want score(question, good) > score(question, bad).
# Build the scoring architecture explicitly: a language-model checkpoint has
# no score.weight, so loading it as a classifier emits a missing-head warning.
# Only the transformer weights transfer; the new scalar head starts random and
# learns from the preference pairs below. Deep-copy the config so changing the
# number of labels cannot modify the policy's configuration.
reward_config = copy.deepcopy(policy.config)
# One output number per input sequence, rather than one score per vocabulary
# token. The new linear layer that produces this number is called the "head".
reward_config.num_labels = 1
reward = AutoModelForSequenceClassification.from_config(reward_config).to(device)
# Copy learned language representations into this independent model. strict=True
# checks that all transformer weight names and shapes match; it does not copy
# GPT-2's language-generation head into the new scoring head.
reward.transformer.load_state_dict(policy.transformer.state_dict(), strict=True)
reward.config.pad_token_id = tok.eos_token_id
reward.train()
# Both the transformer and the new head learn during preference training.
optimizer = torch.optim.AdamW(reward.parameters(), lr=5e-5)


def score(question, response):
    """Return the reward model's raw score for a completed question/answer."""
    x = tokens(prompt(question) + " " + response)
    # logits are raw outputs, without a probability conversion. flatten() changes
    # shape [1, 1] into [1], making score subtraction straightforward below.
    return reward(x, attention_mask=attention_mask(x)).logits.flatten()


# Fifteen passes over four pairs gives 60 preference updates.
for epoch in range(15):
    random.shuffle(examples)
    for question, good, bad in examples:
        # If bad scores higher than good, this difference is positive and the
        # loss is large. Making good score higher makes the loss smaller.
        # softplus(z) = log(1 + exp(z)), implemented stably by PyTorch.
        # Equivalently this is -log(sigmoid(good_score - bad_score)): reward
        # confident preference for the chosen answer. mean() averages the batch.
        loss = F.softplus(score(question, bad) - score(question, good)).mean()
        # Clear old gradients, compute new ones, then update reward-model weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        # Like Step 1, this reports the final pair's loss, not the epoch average.
        print(f"Reward epoch {epoch + 1}: loss={loss.item():.3f}")

# Discard reward training's optimizer state before PPO to free memory.
del optimizer
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
# The judge is now fixed: PPO will change the answer generator, not the judge.
# eval() disables dropout; requires_grad_(False) freezes trainable parameters.
reward.eval().requires_grad_(False)
with torch.no_grad():
    # This checks the SAME pairs used for training. It checks memorization of
    # these preferences, not performance on new questions or unseen answers.
    correct = sum((score(q, good) > score(q, bad)).item()
                  for q, good, bad in examples)
print(f"Reward model prefers the chosen answer in {correct}/{len(examples)} examples")


# 3. PPO: sample an answer, score it, then nudge GPT-2 toward better answers.
# PPO = Proximal Policy Optimization. Unlike Step 1, we do not supply the correct
# answer as a target. The policy writes an answer and the reward model judges it.
# This is a compact teaching version: one sampled answer per step, one scalar
# advantage for its tokens, and a small prompt-based value predictor.
# Production PPO commonly uses larger batches and token-level return estimates.

# Keep three roles separate:
#   policy: the generator we update;
#   reference: a fixed snapshot of the SFT generator, anchoring its behavior;
#   reward: the fixed preference judge trained in Step 2.
policy.eval()  # Turn off dropout so saved log-probabilities are comparable.
# eval() does NOT disable gradients. The policy can still learn in this mode.
# deepcopy makes independent weights; PPO cannot accidentally update reference.
reference = copy.deepcopy(policy).eval().requires_grad_(False)
# A "critic" estimates the reward we expect before seeing the generated answer.
# n_embd is the size of GPT-2's internal token representation; Linear maps that
# vector to one number. This new head starts random and learns alongside policy.
value_head = nn.Linear(policy.config.n_embd, 1).to(device)
# Update policy + critic only. The judge and reference are excluded and frozen.
# PPO uses a smaller learning rate than SFT to make gentler changes.
optimizer = torch.optim.AdamW(
    list(policy.parameters()) + list(value_head.parameters()), lr=1e-6
)


def response_logprobs(model, full, prompt_length):
    """Get log-probability of each generated token under a given model."""
    output = model(full, attention_mask=attention_mask(full))
    # Output at position i predicts token i+1. The final prompt position predicts
    # the first answer token. Drop the final output, which predicts beyond full.
    logits = output.logits[:, prompt_length - 1:-1]
    target = full[:, prompt_length:]
    # log_softmax converts vocabulary scores into log-probabilities. gather picks
    # only the token actually sampled, rather than every possible vocabulary ID.
    # unsqueeze adds the index dimension gather needs; squeeze removes it again.
    # Result shape: [batch size, number of generated tokens].
    return logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)


def policy_stats(full, prompt_length):
    """Return generated-token log-probabilities and the critic's reward guess."""
    # Request internal representations as well as next-token vocabulary scores.
    output = policy(full, attention_mask=attention_mask(full), output_hidden_states=True)
    logits = output.logits[:, prompt_length - 1:-1]
    target = full[:, prompt_length:]
    logp = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    # hidden_states[-1] is the final transformer layer. Select the last PROMPT
    # token, whose causal representation cannot see the future answer. This
    # makes value a baseline for the question, not a score of the completed text.
    value = value_head(output.hidden_states[-1][:, prompt_length - 1]).squeeze(-1)
    return logp, value


for step in range(12):
    # Sample one question. Ignore its supplied answers: the policy now writes one.
    question, _, _ = random.choice(examples)
    x = tokens(prompt(question))
    # Collect a fixed training example without storing gradients for generation,
    # the old policy, reference, or judge. Recompute trainable outputs afterward.
    with torch.no_grad():
        # Sampling explores possible answers. top_p=0.9 keeps a likely-token set
        # covering 90% probability mass; temperature=0.9 sharpens probabilities.
        # full contains BOTH the prompt and the sampled continuation.
        full = policy.generate(x, attention_mask=attention_mask(x), do_sample=True, top_p=0.9, temperature=0.9,
                               max_new_tokens=12, pad_token_id=tok.eos_token_id)
        old_logp, old_value = policy_stats(full, x.shape[1])
        # Evaluate the SAME sampled tokens under the frozen SFT reference.
        ref_logp = response_logprobs(reference, full, x.shape[1])
        rm_score = reward(full, attention_mask=attention_mask(full)).logits.flatten()
        # Sum token log-probability differences to compare whole continuations.
        # Their expectation under the sampling policy is a KL divergence, a
        # measure of distribution change. A single sample can be negative.
        # Subtract this sampled penalty (scaled by 0.02) from the judge's score
        # to discourage drifting too far from the original SFT behavior.
        target_reward = rm_score - 0.02 * (old_logp - ref_logp).sum(dim=1)
        # Advantage = received reward minus expected reward. Positive means
        # "better than expected"; negative means "worse than expected".
        # detach() treats this as a fixed target during the next updates.
        advantage = (target_reward - old_value).detach()

    # Reuse this sampled answer for two updates, keeping old_logp/targets fixed.
    # Clipping removes the incentive for some excessively large probability
    # changes; it is not a hard guarantee that every update stays within 20%.
    for _ in range(2):
        new_logp, value = policy_stats(full, x.shape[1])
        # exp(log(new probability) - log(old probability)) = new / old.
        # A ratio of 1.1 means this token is now 10% more probable.
        ratio = (new_logp - old_logp).exp()
        clipped = ratio.clamp(0.8, 1.2)
        # advantage[:, None] broadcasts the answer's one advantage to its tokens.
        # Take the more conservative of unclipped and clipped improvements.
        # Negate because optimizers minimize losses, while we want more reward.
        policy_loss = -torch.minimum(ratio * advantage[:, None],
                                     clipped * advantage[:, None]).mean()
        # Teach the critic to predict the observed target reward using squared
        # error. A better baseline makes future advantages more informative.
        value_loss = 0.5 * (value - target_reward).square().mean()
        # Jointly train generator and critic, weighting critic error by 0.2.
        loss = policy_loss + 0.2 * value_loss
        optimizer.zero_grad()
        loss.backward()
        # Rescale gradients if their combined norm exceeds 1.0, helping prevent
        # unstable updates. This is separate from PPO's probability-ratio clip.
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_head.parameters()), 1.0
        )
        optimizer.step()
    # Report the judge's raw score BEFORE these updates, without the KL penalty.
    print(f"PPO {step + 1}/12: reward={rm_score.item():.3f}")

# Generate freely after training and save the updated generator for compare.py.
# The separate reward model, reference, and critic are not needed to answer
# questions, so only the policy and tokenizer are saved here.
print("After PPO:", answer(policy, "What is 2 + 2?"))
policy.save_pretrained("gpt2_rlhf_result")
tok.save_pretrained("gpt2_rlhf_result")
print("Saved the resulting language model to ./gpt2_rlhf_result")
