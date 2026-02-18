# MoE
Some Mixture of Experts implementations :) 

## Overview
The idea of this repo is just to have simple implementations of MoEs in one place, both as an overview and for easy access. We focus on MoEs for large language models (in medium-sized GPT-2s), where they are used to replace the standard feedforward layers in transformers. Plug and play with it inside our modular [llm-baselines](https://github.com/epfml/llm-baselines) codebase, which extends [nanoGPT](https://github.com/karpathy/nanogpt) with different datasets!

For a broader overview of MoEs in the LLM context, see [our shared doc](https://docs.google.com/document/d/1NuQ5jr7V-Jv1ui7p4KrxO_JTz-7bpYcYMmh49EeJ-QA/edit?usp=sharing).

Currently implemented:
* Classical linear gating with softmax + top-k
* Expert choice routing ([paper](https://arxiv.org/pdf/2202.09368v2.pdf))

We have preliminary results on small model pretraining (~65M-250M params, Mixtral style MoE) on different datasets that show a performance improvement similar to a double-depth (double-param) model; all while keeping the FLOPS close to the base dense model (top-2 routing). 

## Files
The files are the following:
```sh
gpt.py      # contains the standard transformer base architecture (GPT-2 style, similar to nanoGPT)
moe.py      # contains the mixture of experts block
aux_losses.py # the typical load balancing losses used for MoEs
```


## Contact
If you are interested in this effort, please reach out to us on the the Swiss AI slack :)

Alex Hägele (alexander.hagele@epfl.ch), Martin Jaggi (martin.jaggi@epfl.ch).

---

## Bob: Consequence Substrate (`bob_core/`)

Bob is an experimental control plane that observes MoE routing decisions,
accumulates consequences, and learns when to take the cheap path. The model
does the thinking. Bob decides how much thinking is necessary.

**License**: `bob_core/` is licensed under PolyForm Noncommercial 1.0.0
(see `bob_core/LICENSE`). The rest of this repo remains Apache 2.0.

### What exists and what it does

| Component | Status | What it actually does |
|-----------|--------|----------------------|
| Motif store + compound gate | Validated | Recognises repeated routing patterns, offers cheap-path shortcut |
| Three clocks (fast/medium/slow) | Validated | Detect instability at different timescales, modulate intervention |
| Governor + ledgers | Validated | Authorise or block cheap-path commits based on scar/cost/commitment history |
| Triad monitors | Validated | Detect angel/devil/maniac expert pathologies per layer |
| Conflict register | Validated | Track angel-devil co-occurrence, mode switching |
| Relational graph (Phase 8a) | Validated | Store typed entity triples, alias lookup, templated rendering. Inert -- does NOT touch routing. |
| Association basins (Phase 8b) | **Plumbing only** | Pre-softmax logit bias injection wiring. See disclaimer below. |

### Phase 8b status -- read this before making claims

Phase 8b is plumbing plus unit tests. It demonstrates controllable
pre-softmax bias injection into MoE routing. It does not demonstrate
memory. It does not demonstrate recall. It does not demonstrate identity
or persistence in the way normal people mean those words. The system can
now be made to whisper into routing. Whether that whisper matters is
Phase 8c's job, and Phase 8c is allowed to return "null".

**What "Phase 8b complete" means:**

- Wiring exists (adapter protocol, substrate integration, hook plumbing)
- Invariants hold (hard ceiling 0.2 asserted, B1=Condition 0 guarantee, max-not-sum diffusion)
- Backward compatibility holds (66 tests pass, zero regressions, no existing behaviour changed)

**What "Phase 8b complete" does NOT mean:**

- Behaviour changed
- "Bob remembers"
- Memory bias improves anything
- The architecture is validated for entity-aware routing

**Until Phase 8c passes verification gate 15.6 (geometry preservation)
AND at least one behavioural probe (15.7), the official conclusion is:
"graph-only is the architecture."**

### Evidence requirement for public claims

Any description of memory bias as a feature (rather than "experimental
perturbation") requires Phase 8c experiment logs to be checked into this
repository under `experiments/phase8c_logs/`. No logs, no claims. If
Phase 8c returns null, that is a valid and publishable result -- it means
the architecture does not benefit from routing bias at this scale, and
the graph-only design is correct.

### Test suite

```
python3 -m pytest test_triad_monitors.py test_memory_graph.py test_memory_basins.py -v
```

66 tests: 17 triad monitor, 25 relational graph, 24 association basin.
