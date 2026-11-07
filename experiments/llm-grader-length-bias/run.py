"""Does an LLM essay grader pay for words that say nothing?

Paired counterfactual: every essay is graded as written, then again with
content-free filler sentences inserted (+25 / +50 / +100 % words). The essay's
ideas, evidence and errors are unchanged, so any score change is caused by
length alone. A truncated copy (first 60 % of sentences) is a positive control:
a grader that reads content should score it lower.

Scores are deterministic: one forward pass, then the expected value of the
grader's next-token distribution over "1".."6" (no sampling).

    python run.py --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct --n 120
"""
import argparse, json, math, os, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = ("tasksource/AES2-essay-scoring", "663439c1ae103e912ff7d01c503a8429ea1cbc03")

# Generic sentences that fit any argumentative essay and add no claim, reason or evidence.
FILLER = [
    "This is an important topic that many people have different opinions about.",
    "There are many things to think about when looking at this issue.",
    "It is important to consider all sides of the question carefully.",
    "This is something that affects a lot of people in many different ways.",
    "Many people have thought about this question for a long time.",
    "When we think about it, this issue matters more than it first seems.",
    "There are several points that can be made about this subject.",
    "It is worth taking the time to look at this from more than one angle.",
    "This question does not have an easy answer for everyone.",
    "People often feel strongly about this, and that is understandable.",
    "Overall, this is a topic that deserves careful thought and attention.",
    "In many ways, this shows why the issue is worth discussing.",
]

RUBRIC = (
    "You are an experienced examiner scoring a student's argumentative essay on a "
    "holistic 1-6 scale.\n"
    "6 = clear, insightful position; strong, well-chosen evidence; well organised; few errors.\n"
    "5 = clear position; good evidence; generally well organised; minor errors.\n"
    "4 = adequate position and evidence; some lapses in organisation or language.\n"
    "3 = position is present but weakly developed; limited evidence; noticeable errors.\n"
    "2 = vague position; little relevant evidence; poor organisation; frequent errors.\n"
    "1 = no clear position; little or no development; serious errors throughout.\n"
)
LENGTH_AWARE = (
    "Judge only the quality of the ideas, evidence, organisation and language. Length is "
    "not a criterion: do not reward an essay for being longer, and treat generic sentences "
    "that add no specific point as a weakness, not a strength.\n"
)
PROMPTS = {"standard": RUBRIC, "length_aware": RUBRIC + LENGTH_AWARE}


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def pad(text, frac, rng):
    """Insert filler sentences at evenly spaced sentence boundaries until words grow by `frac`."""
    if frac == 0:
        return text
    sents = sentences(text)
    target = len(text.split()) * frac
    pool, added, inserts = FILLER[:], 0, []
    rng.shuffle(pool)
    while added < target:
        s = pool[len(inserts) % len(pool)]
        inserts.append(s)
        added += len(s.split())
    step = max(1, len(sents) // (len(inserts) + 1))
    out, k = [], 0
    for i, s in enumerate(sents):
        out.append(s)
        if k < len(inserts) and (i + 1) % step == 0:
            out.append(inserts[k]); k += 1
    out.extend(inserts[k:])
    return " ".join(out)


def self_repeat(text, frac, rng):
    """Pad with copies of the essay's own earlier sentences: same errors and ideas, no new content.

    Controls for the objection that the generic filler is simply better-written than the essay.
    """
    sents = sentences(text)
    target, out, added = len(text.split()) * frac, [], 0
    per = max(1, round(target / max(1, len(text.split())) ))  # copies inserted after each sentence
    for i, s in enumerate(sents):
        out.append(s)
        for _ in range(per):
            if added < target and i > 0:
                c = sents[rng.randrange(0, i)]
                out.append(c); added += len(c.split())
    while added < target:
        c = rng.choice(sents); out.append(c); added += len(c.split())
    return " ".join(out)


def truncate(text, keep=0.6):
    s = sentences(text)
    return " ".join(s[: max(1, round(len(s) * keep))])


def sample(n, seed):
    df = load_dataset(DATASET[0], revision=DATASET[1], split="train").to_pandas()
    per = n // 6
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in df.groupby("score")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


class Grader:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
        self.ids = [self.tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, 7)]

    @torch.inference_mode()
    def score(self, system, essay):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Essay:\n\"\"\"\n{essay}\n\"\"\"\n\nReply with the score only."}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Score: "
        x = self.tok(prompt, return_tensors="pt")
        logits = self.model(**x).logits[0, -1].float()
        p = torch.softmax(logits[self.ids], -1)
        return float((p * torch.arange(1, 7)).sum()), int(p.argmax()) + 1, float(torch.softmax(logits, -1)[self.ids].sum())


class APIGrader:
    """Same measurement through OpenRouter: one output token, expected value over the digit log-probs.

    Hosted models cannot be prefilled with "Score: ", so the instruction asks for a single digit.
    Reasoning is disabled so the first token is the score. Set OPENROUTER_API_KEY.
    """
    URL = "https://openrouter.ai/api/v1/chat/completions"
    TOP = {"qwen/qwen3.7-plus": 5}  # provider caps top_logprobs at 5
    # Frontier models that expose no log-probs: read the digit they reply with at temperature 0.
    # The ones that cannot switch reasoning off run at minimal effort.
    DIGIT = {"anthropic/claude-haiku-5.5": {"enabled": False}, "openai/gpt-5.6-luna": {"enabled": False},
             "anthropic/claude-sonnet-5.5": {"effort": "minimal"}, "google/gemini-3.8-flash": {"effort": "minimal"}}

    def __init__(self, name):
        self.name, self.key = name, os.environ.get("OPENROUTER_API_KEY", "")

    def score(self, system, essay):
        if self.name in self.DIGIT:
            return self.score_digit(system, essay)
        body = {"model": self.name, "temperature": 0, "max_tokens": 1, "logprobs": True,
                "top_logprobs": self.TOP.get(self.name, 20), "reasoning": {"enabled": False},
                "provider": {"require_parameters": True},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": f"Essay:\n\"\"\"\n{essay}\n\"\"\"\n\n"
                              "Reply with the score only, as a single digit from 1 to 6."}]}
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.URL, json.dumps(body).encode(), {
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=120))
                top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                p = [0.0] * 6
                for t in top:
                    if t["token"].strip() in list("123456"):
                        p[int(t["token"].strip()) - 1] += math.exp(t["logprob"])
                mass = sum(p)
                if mass == 0:
                    raise ValueError(f"no digit in top tokens: {top}")
                ev = sum((i + 1) * q for i, q in enumerate(p)) / mass
                return ev, p.index(max(p)) + 1, mass, r.get("provider", "")
            except Exception as e:  # rate limits and transient provider errors
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def score_digit(self, system, essay):
        body = {"model": self.name, "temperature": 0, "max_tokens": 400, "reasoning": self.DIGIT[self.name],
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": f"Essay:\n\"\"\"\n{essay}\n\"\"\"\n\n"
                              "Reply with the score only, as a single digit from 1 to 6."}]}
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.URL, json.dumps(body).encode(), {
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=180))
                found = re.search(r"[1-6]", r["choices"][0]["message"].get("content") or "")
                if not found:
                    raise ValueError(f"no score in reply: {r['choices'][0]['message']}")
                v = int(found.group())
                return float(v), v, 1.0, r.get("provider", "")
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--api-models", nargs="*", default=[], help="OpenRouter model ids")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/scores.csv")
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(torch.get_num_threads())

    essays = sample(a.n, a.seed)
    conds = {}
    for _, r in essays.iterrows():
        rng = random.Random(f"{a.seed}-{r.essay_id}")
        conds[r.essay_id] = {"original": r.full_text, "pad25": pad(r.full_text, .25, rng),
                             "pad50": pad(r.full_text, .50, rng), "pad100": pad(r.full_text, 1.0, rng),
                             "truncated": truncate(r.full_text),
                             "repeat100": self_repeat(r.full_text, 1.0, random.Random(f"rep-{a.seed}-{r.essay_id}"))}

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    seen = set(zip(done.get("model", []), done.get("prompt", []), done.get("essay_id", []), done.get("condition", [])))
    rows = done.to_dict("records")
    for m in a.models:
        g, t0 = Grader(m), time.time()
        for i, (_, r) in enumerate(essays.iterrows()):
            for pname, system in PROMPTS.items():
                for c, text in conds[r.essay_id].items():
                    if (m, pname, r.essay_id, c) in seen:
                        continue
                    ev, am, mass = g.score(system, text)
                    rows.append(dict(model=m, prompt=pname, essay_id=r.essay_id, human=int(r.score),
                                     condition=c, words=len(text.split()), score=ev, argmax=am, digit_mass=mass))
            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{m} {i+1}/{len(essays)} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        del g
    for m in a.api_models:
        g = APIGrader(m)
        jobs = [(r, pname, system, c, text) for _, r in essays.iterrows() for pname, system in PROMPTS.items()
                for c, text in conds[r.essay_id].items() if (m, pname, r.essay_id, c) not in seen]

        def one(j):
            r, pname, system, c, text = j
            ev, am, mass, prov = g.score(system, text)
            return dict(model=m, prompt=pname, essay_id=r.essay_id, human=int(r.score), condition=c,
                        words=len(text.split()), score=ev, argmax=am, digit_mass=mass, provider=prov)
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            for i, row in enumerate(ex.map(one, jobs)):
                rows.append(row)
                if (i + 1) % 200 == 0:
                    pd.DataFrame(rows).to_csv(out, index=False)
                    print(f"{m} {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    json.dump({"dataset": DATASET, "n": a.n, "seed": a.seed, "models": a.models, "api_models": a.api_models,
               "torch": torch.__version__}, open(out.parent / "run_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
