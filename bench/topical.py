"""Topical learning curve: does the lexical head earn its keep on topic routing?

Training and test sentences come from disjoint template families. Thresholds are tuned on a separate tune set,
as `ss lock` would; heads settle only at >= 95% precision there; the rest falls back to the judge.
Run from the repository root:
    python -m bench.topical [judge source]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from semantic_switch import Model, Runtime
from semantic_switch.heads import train_lexical
from semantic_switch.tools.lock import tune_side

TOPICS = {"billing": "charges, invoices, payments or refunds", "technical": "the product is broken, erroring or slow",
          "account": "login, password, profile or personal data", "shipping": "delivery and order tracking"}
FILL = {"x": ["the app", "the router", "the dashboard", "the printer", "the website", "o aplicativo", "o roteador"],
        "n": ["4512", "7781", "2203", "9930", "1188"], "m": ["May", "June", "March", "maio", "junho"],
        "a": ["$40", "€19", "R$ 89", "$120"]}
TRAIN = {
    "billing": ["I was charged twice in {m}.", "Please refund the {a} charge.", "My invoice for {m} is wrong.",
                "Why did you bill me {a}?", "Fui cobrado duas vezes em {m}.", "Quero o reembolso de {a}.",
                "A fatura de {m} veio errada.", "Cancel the payment of {a}, it is a mistake."],
    "technical": ["{x} crashes when I open it.", "{x} shows an error since the update.", "{x} is very slow today.",
                  "{x} stopped working.", "{x} trava quando abro.", "{x} dá erro desde a atualização.",
                  "{x} está muito lento hoje.", "{x} parou de funcionar."],
    "account": ["I forgot my password.", "I can't log in to my account.", "Please change my email address.",
                "Delete my profile, please.", "Esqueci minha senha.", "Não consigo entrar na minha conta.",
                "Troquem meu e-mail, por favor.", "Apaguem meu perfil."],
    "shipping": ["Where is order {n}?", "My package {n} has not arrived.", "Tracking for {n} has not moved.",
                 "The parcel was delivered to the wrong house.", "Cadê o pedido {n}?", "Meu pacote {n} não chegou.",
                 "O rastreio do {n} não mudou.", "O pacote foi entregue na casa errada."],
}
TEST = {
    "billing": ["There are two identical debits of {a} on my card statement.", "Can you send me a corrected receipt for {m}?",
                "Tem duas cobranças iguais de {a} no meu cartão.", "Preciso de uma nota fiscal corrigida de {m}.",
                "You took money for a plan I cancelled."],
    "technical": ["Every time I press save, {x} freezes.", "{x} keeps showing a blank white screen.",
                  "Toda vez que salvo, {x} congela.", "{x} fica com a tela branca.", "Nothing loads after the last patch."],
    "account": ["The two-factor code never reaches my phone.", "I want to update the name on my profile.",
                "O código de verificação não chega no celular.", "Quero mudar o nome no meu cadastro.",
                "My username is locked after too many attempts."],
    "shipping": ["The courier says it was left at the door but there's nothing here.", "Order {n} is stuck in customs.",
                 "O entregador diz que deixou na porta, mas não tem nada aqui.", "O pedido {n} está parado na alfândega.",
                 "When will my box get here? It's been ten days."],
}
QUESTION = "What is the message mainly about?"


def sample(templates: dict, per_class: int, rng: random.Random) -> list[tuple[str, int]]:
    keys = list(TOPICS)
    out = []
    for c, topic in enumerate(keys):
        for _ in range(per_class):
            t = rng.choice(templates[topic])
            out.append((t.format(**{k: rng.choice(v) for k, v in FILL.items()}), c))
    rng.shuffle(out)
    return out


def tuned(probs: np.ndarray, y: list[int]) -> float | None:
    return tune_side(list(probs.max(axis=1)), list(probs.argmax(axis=1) == np.array(y)), 0.95, None)[0]


def main():
    rng = random.Random(7)
    test = sample(TEST, 20, rng)
    models = {"judge": Model(sys.argv[1])} if len(sys.argv) > 1 else None
    report = {}
    with Runtime(models) as rt:
        texts = [f"[customer] message: {t}" for t, _ in test]
        ys = [c for _, c in test]
        rt.pick(texts[0], QUESTION, TOPICS)
        t0 = time.perf_counter()
        judged = [rt.pick(t, QUESTION, TOPICS) for t in texts]
        judge_ms = (time.perf_counter() - t0) * 1000 / len(texts)
        keys = list(TOPICS)
        j_pred = [keys.index(r.index) if r.status == "ok" and r.score >= 0.9 else None for r in judged]
        j_wrong = sum(p is not None and p != y for p, y in zip(j_pred, ys))
        report["judge"] = {"accuracy": round(float(np.mean([keys.index(r.index) == y if r.index else False
                                                              for r, y in zip(judged, ys)])), 3),
                           "settled": sum(p is not None for p in j_pred), "wrong": j_wrong, "ms": round(judge_ms, 1)}
        print("judge", report["judge"], flush=True)
        for n in (10, 30, 100):
            train_set = sample(TRAIN, n, rng)
            tune_set = sample(TRAIN, 25, rng)
            tr = [f"[customer] message: {t}" for t, _ in train_set]
            tu = [f"[customer] message: {t}" for t, _ in tune_set]
            ytr, ytu = [c for _, c in train_set], [c for _, c in tune_set]
            sparse = train_lexical(tr, ytr, tuple(keys))
            heads = {"lexical": (np.array([sparse.probs(t) for t in tu]), None, None)}
            t0 = time.perf_counter()
            lex_test = np.array([sparse.probs(t) for t in texts])
            heads["lexical"] = (heads["lexical"][0], lex_test, (time.perf_counter() - t0) * 1000 / len(texts))
            for kind, (p_tune, p_test, ms) in heads.items():
                th = tuned(p_tune, ytu)
                settle = [int(p.argmax()) if th is not None and p.max() >= th else None for p in p_test]
                head_wrong = sum(s is not None and s != y for s, y in zip(settle, ys))
                cascade = [s if s is not None else j for s, j in zip(settle, j_pred)]
                fallback = sum(s is None for s in settle) / len(ys)
                row = {"head_accuracy": round(float(np.mean(p_test.argmax(axis=1) == np.array(ys))), 3),
                       "threshold": th, "head_settled": sum(s is not None for s in settle), "head_wrong": head_wrong,
                       "cascade_settled": sum(c is not None for c in cascade),
                       "cascade_wrong": sum(c is not None and c != y for c, y in zip(cascade, ys)),
                       "judge_calls": round(fallback, 3), "cascade_ms": round(ms + fallback * judge_ms, 1)}
                report[f"{kind}@{n}"] = row
                print(f"{kind}@{n}", row, flush=True)
    out = Path("bench/results/topical.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
