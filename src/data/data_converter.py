"""Convert raw OpenR1-Math examples into InftyThink-format training instances."""
from __future__ import annotations
import numpy as np
from transformers import PreTrainedTokenizer

from src.data.segmenter import segment_trace
from src.data.summary_generator import heuristic_summary

# Special control tokens injected into the vocabulary
CONTROL_TOKENS = ["<SEGMENT>", "<SUMMARY>", "<FINAL>"]

# Task label constants
TASK_SEGMENT = "segment"
TASK_SUMMARY = "summary"
TASK_FINAL = "final"


def build_segment_input(
    problem: str,
    summaries: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
) -> dict:
    """Build the input for a segment-generation step.

    Format: [problem] <SUMMARY_1> ... <SUMMARY_k> <SEGMENT>
    """
    parts = [problem.strip()]
    for s in summaries:
        parts.append("<SUMMARY> " + s.strip())
    parts.append("<SEGMENT>")
    text = "\n".join(parts)
    return _encode(text, tokenizer, max_length)


def build_summary_input(
    problem: str,
    summaries: list[str],
    latest_segment: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
) -> dict:
    """Build the input for a summary-generation step.

    Format: [problem] <SUMMARY_1>...<SUMMARY_k> [latest_segment] <SUMMARY>
    """
    parts = [problem.strip()]
    for s in summaries:
        parts.append("<SUMMARY> " + s.strip())
    parts.append(latest_segment.strip())
    parts.append("<SUMMARY>")
    text = "\n".join(parts)
    return _encode(text, tokenizer, max_length)


def build_final_input(
    problem: str,
    summaries: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
) -> dict:
    """Build the input for the final-answer step.

    Format: [problem] <SUMMARY_1>...<SUMMARY_T> <FINAL>
    """
    parts = [problem.strip()]
    for s in summaries:
        parts.append("<SUMMARY> " + s.strip())
    parts.append("<FINAL>")
    text = "\n".join(parts)
    return _encode(text, tokenizer, max_length)


def convert_example(
    problem: str,
    trace: str,
    answer: str,
    tokenizer: PreTrainedTokenizer,
    segment_len: int = 128,
    summary_len: int = 32,
    max_seq_len: int = 1024,
) -> list[dict]:
    """Convert one problem+trace+answer into a list of training instances.

    Returns a list of dicts, one per (segment | summary | final) step:
        {
            "input_ids":   np.ndarray  shape (max_seq_len,)  dtype int32
            "target_ids":  np.ndarray  shape (max_seq_len,)  dtype int32
            "loss_mask":   np.ndarray  shape (max_seq_len,)  dtype float32
                             1.0 on target tokens, 0.0 on input tokens
            "task":        str  "segment" | "summary" | "final"
            "step":        int
            "n_steps":     int
        }
    """
    segments = segment_trace(trace, tokenizer, segment_len=segment_len)
    if not segments:
        return []

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    instances: list[dict] = []
    running_summaries: list[str] = []
    n_steps = 2 * len(segments) + 1  # seg, sum, seg, sum, ..., final

    for i, seg in enumerate(segments):
        # --- segment step ---
        inp = build_segment_input(problem, running_summaries, tokenizer, max_seq_len)
        tgt = _encode(seg, tokenizer, segment_len)
        instances.append(_make_instance(inp, tgt, TASK_SEGMENT, 2 * i, n_steps, max_seq_len, pad_id))

        # --- summary step ---
        summary = heuristic_summary(seg, tokenizer, max_summary_tokens=summary_len)
        inp_s = build_summary_input(problem, running_summaries, seg, tokenizer, max_seq_len)
        tgt_s = _encode(summary, tokenizer, summary_len)
        instances.append(_make_instance(inp_s, tgt_s, TASK_SUMMARY, 2 * i + 1, n_steps, max_seq_len, pad_id))

        running_summaries.append(summary)

    # --- final answer step ---
    inp_f = build_final_input(problem, running_summaries, tokenizer, max_seq_len)
    tgt_f = _encode(answer, tokenizer, 128)
    instances.append(_make_instance(inp_f, tgt_f, TASK_FINAL, n_steps - 1, n_steps, max_seq_len, pad_id))

    return instances


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode(text: str, tokenizer: PreTrainedTokenizer, max_length: int) -> dict:
    """Encode text, truncate to max_length, return {"input_ids": np.array}."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:max_length]
    return {"input_ids": np.array(ids, dtype=np.int32)}


def _make_instance(
    inp: dict,
    tgt: dict,
    task: str,
    step: int,
    n_steps: int,
    max_seq_len: int,
    pad_id: int = 0,
) -> dict:
    """Pack input + target into a padded training instance with a loss mask.

    The sequence is: [input_ids ... target_ids ... <pad>]
    loss_mask is 1.0 only on target token positions.
    """
    inp_ids = inp["input_ids"]
    tgt_ids = tgt["input_ids"]

    # Truncate input to leave room for target
    max_inp = max_seq_len - len(tgt_ids)
    if max_inp < 0:
        max_inp = 0
    inp_ids = inp_ids[-max_inp:] if len(inp_ids) > max_inp else inp_ids

    combined = np.concatenate([inp_ids, tgt_ids])
    loss_mask = np.concatenate([
        np.zeros(len(inp_ids), dtype=np.float32),
        np.ones(len(tgt_ids), dtype=np.float32),
    ])

    # Pad to max_seq_len
    pad_len = max_seq_len - len(combined)
    if pad_len > 0:
        combined = np.concatenate([combined, np.full(pad_len, pad_id, dtype=np.int32)])
        loss_mask = np.concatenate([loss_mask, np.zeros(pad_len, dtype=np.float32)])
    else:
        combined = combined[:max_seq_len]
        loss_mask = loss_mask[:max_seq_len]

    # Targets = shift combined left by 1 (next-token prediction)
    target_ids = np.concatenate([combined[1:], np.array([pad_id], dtype=np.int32)])

    return {
        "input_ids": combined,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "task": task,
        "step": step,
        "n_steps": n_steps,
    }
