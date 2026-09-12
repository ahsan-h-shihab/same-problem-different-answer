"""Evaluate a frontier LLM on the fixed ~300-item XNLI subset, preserving the 15-language
parallel structure. Identical prompt template for every (item, language, model); only the
premise/hypothesis language varies. All raw responses are cached to JSONL so scoring is
fully reproducible from committed outputs (hosted LLMs are never bit-reproducible on re-query).

Providers: openrouter (used for all three reported models), plus openai / anthropic / gemini
back-ends (see the paper, Appendix B, for which provider served which calls).
Output: results/preds__{tag}__test300.npz  (preds[300,15] int8, -1=unparseable; gold[300]; indices)
        results/llm_raw__{tag}.jsonl        (every raw response, keyed by idx+lang)
"""
import os, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xnli_load
from xnli_load import LANGS

RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(__file__), "..", "results")
LABELS = ["entailment", "neutral", "contradiction"]  # id 0,1,2

SYS = ("You are an expert linguistic annotator for Natural Language Inference. "
       "Given a premise and a hypothesis, decide the relationship between them. "
       "Answer with exactly one word, choosing from: entailment, neutral, or contradiction. "
       "Output only that single word, with no explanation and no punctuation.")


def user_prompt(premise, hypothesis):
    return (f"Premise: {premise}\nHypothesis: {hypothesis}\n"
            "Relationship (entailment, neutral, or contradiction):")


def parse_label(text):
    if text is None:
        return -1
    t = text.strip().lower()
    # exact / prefix / substring match, in priority order
    for i, lab in enumerate(LABELS):
        if t == lab:
            return i
    for i, lab in enumerate(LABELS):
        if t.startswith(lab):
            return i
    # handle stray punctuation/quotes
    t2 = "".join(ch for ch in t if ch.isalpha())
    for i, lab in enumerate(LABELS):
        if lab in t2:
            return i
    # common shorthand
    if t2 in ("entail", "entails"):
        return 0
    if t2 in ("contradict", "contradicts", "contra"):
        return 2
    return -1


# ---- provider back-ends -------------------------------------------------------
class OpenAIBackend:
    def __init__(self, model, temperature, reasoning_effort="minimal"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort  # NLI label needs no deliberation; keeps output non-empty & cheap

    def __call__(self, sys, usr):
        # Reasoning models spend hidden tokens against the budget -> give ample headroom
        # so the visible one-word answer is never truncated to empty.
        kw = dict(model=self.model,
                  messages=[{"role": "system", "content": sys},
                            {"role": "user", "content": usr}],
                  max_completion_tokens=256, seed=42)
        if self.temperature is not None and abs(self.temperature - 1.0) > 1e-9:
            kw["temperature"] = self.temperature
        try:
            r = self.client.chat.completions.create(reasoning_effort=self.reasoning_effort, **kw)
        except Exception as e:
            if "reasoning_effort" in str(e):
                r = self.client.chat.completions.create(**kw)
            else:
                raise
        return r.choices[0].message.content


class AnthropicBackend:
    def __init__(self, model, temperature, reasoning_effort="low"):
        import anthropic
        from keys import load_key
        self.client = anthropic.Anthropic(api_key=load_key("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature

    def __call__(self, sys, usr):
        r = self.client.messages.create(
            model=self.model, system=sys, max_tokens=16,
            temperature=(1.0 if self.temperature is None else self.temperature),
            messages=[{"role": "user", "content": usr}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenRouterBackend:
    """OpenAI-compatible transport for OpenRouter (Claude, Gemini, GPT-5.6, etc.).
    reasoning_effort is passed via extra_body when supported; falls back gracefully."""
    def __init__(self, model, temperature, reasoning_effort="low"):
        from openai import OpenAI
        from keys import load_key
        key = load_key("OPENROUTER_API_KEY", "openrouter.key")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        self.model = model
        self.temperature = 1.0 if temperature is None else temperature
        self.reasoning_effort = reasoning_effort

    def __call__(self, sys, usr):
        kw = dict(model=self.model,
                  messages=[{"role": "system", "content": sys},
                            {"role": "user", "content": usr}],
                  max_tokens=256, temperature=self.temperature)
        # ask reasoning models for minimal thinking so the one-word answer isn't truncated
        try:
            r = self.client.chat.completions.create(
                extra_body={"reasoning": {"effort": self.reasoning_effort}}, **kw)
        except Exception as e:
            if "reasoning" in str(e).lower():
                r = self.client.chat.completions.create(**kw)
            else:
                raise
        msg = r.choices[0].message
        return msg.content if msg and msg.content else ""


class GeminiBackend:
    """Google Gemini via its OpenAI-compatible endpoint, using GEMINI_API_KEY directly.
    reasoning_effort='low' keeps a thinking model from burning the budget before the label."""
    def __init__(self, model, temperature, reasoning_effort="low"):
        from openai import OpenAI
        from keys import load_key
        key = load_key("GEMINI_API_KEY")
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=key)
        self.model = model
        self.temperature = 1.0 if temperature is None else temperature
        self.reasoning_effort = reasoning_effort

    def __call__(self, sys, usr):
        kw = dict(model=self.model,
                  messages=[{"role": "system", "content": sys},
                            {"role": "user", "content": usr}],
                  max_tokens=512, temperature=self.temperature)
        try:
            r = self.client.chat.completions.create(reasoning_effort=self.reasoning_effort, **kw)
        except Exception as e:
            if "reasoning" in str(e).lower():
                r = self.client.chat.completions.create(**kw)
            else:
                raise
        msg = r.choices[0].message
        return msg.content if msg and msg.content else ""


BACKENDS = {"openai": OpenAIBackend, "anthropic": AnthropicBackend,
            "openrouter": OpenRouterBackend, "gemini": GeminiBackend}


CREDIT_SIGNALS = ("insufficient credit", "insufficient_quota",
                  "exceeded your current quota", "payment required")


def is_credit_error(msg):
    m = (msg or "").lower()
    return any(s in m for s in CREDIT_SIGNALS)


def call_with_retry(backend, sys, usr, max_retries=10):
    delay = 3.0
    for attempt in range(max_retries):
        try:
            return backend(sys, usr), None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
            # fail fast on credit/quota exhaustion: retrying cannot help until a top-up
            if is_credit_error(str(e)):
                return None, err
            if attempt == max_retries - 1:
                return None, err
            d = delay * (2.0 if "RateLimit" in type(e).__name__ else 1.0)
            time.sleep(min(d, 90))
            delay = min(delay * 1.7, 90)
    return None, "exhausted"


def _ok(r):
    o = r.get("output")
    return r.get("error") is None and o is not None and str(o).strip() != ""


def load_cache(path):
    """Return best record per (idx,lang), preferring a successful call over a failed one,
    so transient failures (e.g. a network outage) are retried rather than frozen as invalid."""
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (r["idx"], r["lang"])
            if key not in cache or (_ok(r) and not _ok(cache[key])):
                cache[key] = r
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(BACKENDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short id for filenames, e.g. gpt56 / sonnet5")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning_effort", default="low", help="minimal/low/medium/high where supported")
    ap.add_argument("--limit_items", type=int, default=0, help="debug: only first K subset items")
    ap.add_argument("--limit_langs", type=int, default=0, help="debug: only first K langs")
    args = ap.parse_args()

    sub = json.load(open(os.path.join(RESULTS, "llm_subset.json")))
    idxs = sub["indices"]
    if args.limit_items:
        idxs = idxs[:args.limit_items]
    langs = LANGS[:args.limit_langs] if args.limit_langs else LANGS

    data, gold = xnli_load.load_all("test")
    gold_sub = np.array([gold[i] for i in idxs], dtype=np.int8)

    raw_path = os.path.join(RESULTS, f"llm_raw__{args.tag}.jsonl")
    cache = load_cache(raw_path)
    lock = threading.Lock()
    raw_f = open(raw_path, "a", encoding="utf-8", newline="\n")
    backend = BACKENDS[args.provider](args.model, args.temperature, args.reasoning_effort)

    jobs = []
    for ii, i in enumerate(idxs):
        for lj, lg in enumerate(langs):
            c = cache.get((i, lg))
            if c is not None and _ok(c):
                continue  # only skip successful calls; retry failed/empty ones
            jobs.append((ii, i, lj, lg))
    print(f"[{args.tag}] {len(idxs)} items x {len(langs)} langs; {len(jobs)} new calls "
          f"({len(cache)} cached)", flush=True)

    done = [0]
    credit_out = threading.Event()  # set on first credit/quota error -> abort the run

    def work(job):
        ii, i, lj, lg = job
        if credit_out.is_set():
            return None  # short-circuit: don't burn calls once credit is exhausted
        prem, hyp = data[lg][0][i], data[lg][1][i]
        out, err = call_with_retry(backend, SYS, user_prompt(prem, hyp))
        if err and is_credit_error(err):
            credit_out.set()
            with lock:
                print(f"[{args.tag}] CREDIT EXHAUSTED at {done[0]} calls: {err[:120]}", flush=True)
            return None  # do not cache credit failures (retry after top-up)
        rec = {"idx": int(i), "lang": lg, "output": out, "error": err}
        with lock:
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_f.flush()
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"[{args.tag}] {done[0]}/{len(jobs)} calls", flush=True)
        return rec

    if jobs:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for _ in as_completed([ex.submit(work, j) for j in jobs]):
                pass
    raw_f.close()
    if credit_out.is_set():
        print(f"[{args.tag}] ABORTED: API credit exhausted. Re-run after top-up to resume.", flush=True)

    # assemble predictions from the (now complete) cache
    cache = load_cache(raw_path)
    preds = np.full((len(idxs), len(langs)), -1, dtype=np.int8)
    n_invalid = 0
    for ii, i in enumerate(idxs):
        for lj, lg in enumerate(langs):
            rec = cache.get((i, lg))
            lab = parse_label(rec["output"]) if rec else -1
            if lab < 0:
                n_invalid += 1
            preds[ii, lj] = lab
    np.savez_compressed(os.path.join(RESULTS, f"preds__{args.tag}__test300.npz"),
                        preds=preds, gold=gold_sub, indices=np.array(idxs), langs=np.array(langs))
    valid = preds >= 0
    acc = float((preds[valid] == gold_sub[:, None].repeat(len(langs), 1)[valid]).mean())
    print(f"[{args.tag}] DONE. invalid={n_invalid}/{preds.size} "
          f"({100*n_invalid/preds.size:.1f}%)  overall_acc(valid)={acc:.3f}", flush=True)

    # record exact model identity + settings for the reproducibility appendix
    meta_path = os.path.join(RESULTS, "llm_models.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta[args.tag] = {"provider": args.provider, "model": args.model,
                      "temperature": args.temperature,
                      "reasoning_effort": args.reasoning_effort, "n_items": len(idxs),
                      "n_langs": len(langs), "invalid": int(n_invalid),
                      "invalid_frac": float(n_invalid / preds.size)}
    json.dump(meta, open(meta_path, "w", encoding="utf-8", newline="\n"), indent=2)


if __name__ == "__main__":
    main()
