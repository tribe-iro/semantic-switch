"""Can the judge tell "false" from "not stated"? Forced true/false vs a three-way pick with an explicit not-stated option.

Run from the repository root:
    python -m bench.unstated ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf
"""
import sys

from semantic_switch import Model
from bench.outcomes import runtime

CLAIM = "There is a visible hydraulic leak."
CASES = [
    ("MX-120A reports E17 after restarting. Fluid is pooling under the pump.", "true"),
    ("MX-120A reports E17 after restarting. No visible leak.", "false"),
    ("MX-120A reports E17 after restarting.", "unstated"),
    ("E17 on the MX-120A again. Floor is dry, nothing dripping.", "false"),
    ("E17 on the MX-120A again. Operator went home early.", "unstated"),
    ("A MX-120A deu E17 ao religar. Há óleo escorrendo da bomba.", "true"),
    ("A MX-120A deu E17 ao religar. Sem vazamento visível.", "false"),
    ("A MX-120A deu E17 ao religar.", "unstated"),
    ("I didn't check for leaks yet; E17 after restart.", "unstated"),
    ("Checked for leaks: none. E17 after restart.", "false"),
]
THREE = {"true": "the input states it is true", "false": "the input states it is false",
         "unstated": "the input does not say"}


def main():
    with runtime(Model(sys.argv[1], adapter="chat")) as rt:
        rt.check("warm", "x")
        forced = three = 0
        for text, want in CASES:
            c = rt.check(f"[technician] report: {text}", CLAIM)
            p = rt.pick(f"[technician] report: {text}", f"Claim: {CLAIM} What does the input say about this claim?", THREE)
            forced += want != "unstated" and c.value == (want == "true")
            three += p.index == want
            print(f"{want:9} check p_true={c.p_true:.2f}  three-way={p.index:9} {p.score:.2f}  | {text[:48]}")
        print(f"forced true/false: {forced}/7 on stated items (3 unstated items get a forced guess); three-way: {three}/10")


if __name__ == "__main__":
    main()
