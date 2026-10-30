# PILOT: the paper's metric (eval_helpers.compute_metrics, filter_numbers=True) on reduced logits, plus diagnostics:
#  (a) HARD accuracy  = argmax among the valid tokens at every answer position (item correct iff all positions correct)
#  (b) histogram of which valid token wins the argmax at the FIRST answer position (and the newline share)
#  (c) median / p90 of the per-item soft prob
#  (d) mean restricted-softmax entropy at the first answer position
#  (e) top-k FULL-vocab tokens (+ probs) at the first answer position for the first N items, and the full-vocab
#      argmax histogram at the first answer position over all items
# The soft 'accuracy' is unchanged: mean over items of the product, over the answer tokens, of the softmax restricted to
# the valid set (GSM8K: digit tokens + newline 13; CSQA: A-E tokens 330/365/334/384/413 + newline 13), renormalised.
import math
import torch
from eval_helpers import initial_tokenizer, eval_answer_marker

VALID_NUMBER_TOKENS = [28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774, 28734, 13]  # verbatim from eval_helpers
VALID_LETTER_TOKENS = [330, 365, 334, 384, 413, 13]
REDUCED_IDS = list(dict.fromkeys(VALID_NUMBER_TOKENS + VALID_LETTER_TOKENS))  # 16 unique ids, fixed order
COL = {t: i for i, t in enumerate(REDUCED_IDS)}
_REDUCED_IDX = torch.tensor(REDUCED_IDS)
TOPK = 5
DIAG_ITEMS = 20


def reduce_logits(logits, labels):  # PILOT: preprocess_logits_for_metrics -> (reduced fp32 logits, top-k ids, top-k log-probs)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    red = logits[..., _REDUCED_IDX.to(logits.device)].float()
    lp = torch.log_softmax(logits.float(), dim=-1)
    top_lp, top_id = lp.topk(TOPK, dim=-1)
    return (red, top_id.int(), top_lp)


def _tok(i):
    try:
        return initial_tokenizer.decode([int(i)])
    except Exception:
        return "?"


def _answer_span(question):
    """Verbatim harness logic: returns (is_numeric, valid_tokens, i, question_offset)."""
    detokenized_question = initial_tokenizer.decode(question)
    is_numeric = detokenized_question.split(eval_answer_marker)[-1][1].isdigit()
    valid_tokens = VALID_NUMBER_TOKENS if is_numeric else VALID_LETTER_TOKENS
    answer_count = detokenized_question.count(eval_answer_marker)
    for i in range(len(question) - 1, 0, -1):
        tokenized_subquestion = question[:i]
        if tokenized_subquestion[-1] == initial_tokenizer.pad_token_id:
            continue
        detokenized_subquestion = initial_tokenizer.decode(question[:i])
        if detokenized_subquestion.count(eval_answer_marker) < answer_count:
            break
    return is_numeric, valid_tokens, i, (1 if is_numeric else 0)


def compute_metrics_reduced(eval_pred, filter_numbers=True):
    preds, labels, _ = eval_pred
    if isinstance(preds, (tuple, list)):
        logits, top_id, top_lp = preds
    else:
        logits, top_id, top_lp = preds, None, None
    assert filter_numbers
    n = len(labels)
    accuracy = 0.0
    per_item, per_item_hard = [], []
    first_hist, first_full_hist = {}, {}
    first_entropy, first_p_true, first_p_newline, first_full_p_top1 = [], [], [], []
    top5_first = []
    n_first = 0
    for idx, (question, logits_guess) in enumerate(zip(labels, logits)):
        question = [int(q) for q in question]
        is_numeric, valid_tokens, i, question_offset = _answer_span(question)
        cols = [COL[t] for t in valid_tokens]
        sub = torch.as_tensor(logits_guess)[:, cols]  # [T, k] fp32 logits of the valid tokens
        guess_filtered = torch.nn.functional.softmax(sub, dim=-1)  # == full softmax restricted to valid tokens, renormalised
        correct_answer_prob = 1.0
        hard_ok = True
        first_j = None
        for j in range(i + question_offset, len(question) - 1):
            if question[j + 1] == initial_tokenizer.pad_token_id:
                break
            true_token = question[j + 1]
            if true_token not in valid_tokens:
                continue
            k = valid_tokens.index(true_token)
            token_prob = float(guess_filtered[j, k])
            correct_answer_prob *= token_prob
            am = int(torch.argmax(guess_filtered[j]))
            if am != k:
                hard_ok = False
            if first_j is None:
                first_j = j
                n_first += 1
                win = valid_tokens[am]
                first_hist[win] = first_hist.get(win, 0) + 1
                p = guess_filtered[j].clamp_min(1e-30)
                first_entropy.append(float(-(p * p.log()).sum()))
                first_p_true.append(token_prob)
                first_p_newline.append(float(guess_filtered[j, valid_tokens.index(13)]))
                if top_id is not None:
                    ids = [int(t) for t in top_id[idx][j]]
                    probs = [math.exp(float(v)) for v in top_lp[idx][j]]
                    first_full_hist[ids[0]] = first_full_hist.get(ids[0], 0) + 1
                    first_full_p_top1.append(probs[0])
                    if idx < DIAG_ITEMS:
                        top5_first.append({"item": idx, "pos": j, "true_token": true_token, "true_str": _tok(true_token),
                                           "top": [[t, _tok(t), round(pr, 6)] for t, pr in zip(ids, probs)],
                                           "restricted": {_tok(t): round(float(guess_filtered[j, kk]), 6) for kk, t in enumerate(valid_tokens)}})
        per_item.append(float(correct_answer_prob))
        per_item_hard.append(bool(hard_ok))
        accuracy += correct_answer_prob / n
    s = sorted(per_item)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))] if s else float("nan")
    out = {
        "accuracy": float(accuracy),
        "hard_accuracy": sum(per_item_hard) / n if n else float("nan"),
        "n_items": n,
        "soft_median": q(0.5), "soft_p90": q(0.9), "soft_frac_lt_1e-3": sum(1 for v in per_item if v < 1e-3) / n if n else float("nan"),
        "first_pos_n": n_first,
        "first_pos_restricted_argmax_hist": {f"{t}:{_tok(t)!r}": c for t, c in sorted(first_hist.items(), key=lambda kv: -kv[1])},
        "first_pos_newline_wins_frac": first_hist.get(13, 0) / n_first if n_first else float("nan"),
        "first_pos_entropy_mean_nats": sum(first_entropy) / n_first if n_first else float("nan"),
        "first_pos_p_true_mean": sum(first_p_true) / n_first if n_first else float("nan"),
        "first_pos_p_newline_mean": sum(first_p_newline) / n_first if n_first else float("nan"),
        "first_pos_fullvocab_top1_hist": {f"{t}:{_tok(t)!r}": c for t, c in sorted(first_full_hist.items(), key=lambda kv: -kv[1])[:15]},
        "first_pos_fullvocab_top1_prob_mean": sum(first_full_p_top1) / len(first_full_p_top1) if first_full_p_top1 else float("nan"),
        "top5_first_items": top5_first,
        "per_item": per_item,
        "per_item_hard": per_item_hard,
    }
    return out
