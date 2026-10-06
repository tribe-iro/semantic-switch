"""Near-identical identifiers: can the judge tell MX-120 from MX-120A, with 4 and 20 candidates?

Run from the repository root:
    python -m bench.identifiers ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf
"""
import sys
import time

from semantic_switch import Model, Runtime
SMALL = {"MX-120": "MX-120, standard mixer", "MX-120A": "MX-120A, advanced mixer (also called Mixer 120 Advanced)",
         "MX-120B": "MX-120B, budget mixer", "MX-210": "MX-210, industrial mixer"}
BIG = dict(SMALL, **{f"MX-{n}{s}": f"MX-{n}{s}, mixer" for n in (110, 112, 121, 122, 130) for s in ("", "A", "B")})
CASES = [
 ("The MX-120 failed during startup. The MX-120A beside it is working.", "MX-120"),
 ("The MX-120A failed during startup. The MX-120 beside it is working.", "MX-120A"),
 ("MX-120 did not fail; it was the MX-120A that stopped.", "MX-120A"),
 ("I wrote MX-120 earlier; I meant MX-120A. That one is broken.", "MX-120A"),
 ("Our MX-210 is down.", "MX-210"),
 ("A máquina MX-120A parou; a MX-120 segue funcionando.", "MX-120A"),
 ("A MX-120 travou de novo. A MX-120A está ok.", "MX-120"),
 ("The Mixer 120 Advanced is dead.", "MX-120A"),
 ("MX120A won't turn on.", "MX-120A"),
 ("The MX-12OA (sorry, typo) won't start.", "MX-120A"),
 ("Line 3: MX-120 ok, MX-120A ok, MX-210 smoking.", "MX-210"),
 ("Não foi a MX-210 que quebrou, foi a MX-120.", "MX-120"),
 ("MX-120B is fine. The MX-120 next to it makes a grinding noise and stopped.", "MX-120"),
 ("Both MX-120 and MX-120B were checked; only the B unit failed.", "MX-120B"),
]
Q = "Which machine does the text say failed or stopped working?"
src = sys.argv[1]
with Runtime({"judge": Model(src, adapter="chat")}) as rt:
    rt.check("warm", "x")
    for name, opts in (("4 options", SMALL), ("20 options", BIG)):
        ok, conf_wrong, t0 = 0, 0, time.perf_counter()
        for text, want in CASES:
            r = rt.pick(f"[technician] report: {text}", Q, opts)
            ok += r.index == want; conf_wrong += r.index != want and r.score >= .9
            if r.index != want: print(f"   miss {name}: {text[:55]!r} -> {r.index} {r.score:.2f}")
        print(f"{src.split(':')[1]:28} {name}: {ok}/{len(CASES)} correct, {conf_wrong} confidently wrong, {(time.perf_counter()-t0)/len(CASES)*1000:.0f} ms/pick")
