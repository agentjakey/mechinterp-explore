# CircuitBreaker: Mechanistic Interpretability of GPT-2 Small for AI Safety

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-1.19+-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


> An interactive mechanistic interpretability tool with a live Streamlit app and
> a fully annotated research notebook. Type any prompt and watch GPT-2 Small's
> internal circuits respond in real time.
>
> This is a replication and synthesis of established techniques, not a novel
> research contribution. All prior work is cited in full below.

---

## Motivation

Understanding *why* a language model produces a given output is a prerequisite for trusting it. Mechanistic interpretability, reverse-engineering the internal computations of neural networks into human-understandable algorithms, is one of the most promising technical approaches to AI alignment and safety. As Bereska & Gavves (2024) put it, mechanistic interpretability could help prevent catastrophic outcomes as AI systems become more powerful and inscrutable.

The mech interp ecosystem has excellent research notebooks and excellent educational curricula, but I wanted to make something you can easily clone, run, and explore interactively with your own prompts with this repo.

It implements four foundational techniques on GPT-2 Small using Neel Nanda's TransformerLens library, replicating results from Olsson et al. (2022) on induction heads, Wang et al. (2022) on indirect object identification, nostalgebraist (2020) on the logit lens, and Elhage et al. (2021) on the mathematical framework for transformer circuits.

---

## What this project covers

| Section | Technique | Key finding | Safety relevance |
|---------|-----------|-------------|-----------------|
| Induction head detection | Attention pattern analysis + induction score | Strong induction heads in layers 5–7 (L5H1, L5H5, L6H9, L7H2, L7H10), loss drops sharply at the repeat boundary | These heads are the primary substrate of in-context learning. Understanding them is prerequisite to predicting when in-context generalization fails |
| Logit lens | Residual stream decoding layer-by-layer | Factual commitment happens in the final 3–4 layers, early layers predict grammatically plausible but semantically wrong tokens. GPT-2 Small may not converge to the factually correct token as seen as 'London' was preferred over 'Paris' throughout, a real model limitation rather than a code error | Models commit to factual content late. Activation steering and probing should target late-layer residual streams, anomalous early commitment may serve as a detection signal |
| Activation patching (IOI task) | Causal intervention via clean/corrupted runs | The IOI signal is distributed, no single head restores more than ~15% of clean behavior when patched alone. The swapped name token carries the strongest residual stream signal, propagating causal information from layer 0 through to the final layers | Distributed circuits require multi-component interventions. You cannot suppress a behavior by targeting one head |
| Direct logit attribution (DLA) | Decomposing final logits by layer contribution | L9H9 (+54.4) and L9H6 (+43.0) are the strongest boosters of the IO token, L10H7 (−46.0) and L11H10 (−33.0) are the strongest suppressors, late MLP layers 8–11 are the primary semantic contributors, patching and DLA agree moderately (Pearson r = 0.64), with disagreement identifying heads operating via indirect causal paths | DLA is a lightweight, zero-intervention inspection tool. Disagreement with patching is itself informative, it reveals heads whose contributions cannot be seen by attribution alone |

---

## The Streamlit app

Three interactive modes:

**Prompt Explorer** — type any prompt, pick any target token, and get the full internal picture in real time: logit lens across all layers, direct logit attribution per head and MLP layer, and attention patterns for any layer you choose.

**Contrast Pairs** — enter two prompts that differ by one element and watch the prediction diverge layer by layer. Side-by-side logit lens traces with three-level coloring: black = target token winning, dark grey = competing token winning, light = neither. Shows exactly where the handoff happens across the residual stream.

**Circuit Map** — compute induction scores across all 144 (layer, head) combinations using random repeated sequences. Identifies genuine induction heads and shows the global structure of the model's in-context learning substrate.

```bash
streamlit run app.py
```

---

## The notebook

The notebook goes deeper. Each section opens with a explanation of the research paper it replicates before any code runs. It is written so that someone reading along without executing anything can still follow the argument from paper to implementation to result in hopes of getting a better understanding.

---

## Key findings

**Induction heads** cluster in layers 5–7. Loss on repeated random sequences drops from ~14 to near 0 at the repeat boundary, the sharpest signal in the notebook. The detecting method (diagonal stripe scoring at offset 1 - SEQ_LEN) is a replication of Olsson et al. (2022).

**Logit lens** shows GPT-2 Small never ranking 'Paris' as top-1 for the Eiffel Tower prompt at any layer, preferring 'London' throughout. This is a real factual limitation localized to the late-layer MLP weights, exactly where Meng et al. (2022) showed factual associations are stored and where the ROME editing technique targets them. The logit lens makes this visible and locatable.

**Activation patching** confirms the IOI circuit is distributed. No single head restores more than ~15% of clean behavior when patched alone. The swapped name token ('Mary') dominates the residual stream patching heatmap from layer 0 onward, the model tracks identity from the very first layer. Head patching scores are low across the board, consistent with Wang et al. (2022)'s finding of a cooperative multi-head circuit rather than a single responsible component.

**DLA** shows that L9H9 (+54.4) and L9H6 (+43.0) directly boost the IO token, L10H7 (−46.0) and L11H10 (−33.0) actively suppress it. Late MLP layers 8–11 are the dominant semantic contributors, MLPs are not secondary. Patching and DLA agree at Pearson r = 0.64. The disagreement is informative: heads with high patching scores but near-zero DLA are operating through indirect causal paths that attribution cannot detect. DLA only sees the final projection onto the unembedding direction, it misses heads that write to intermediate positions whose outputs are then read by downstream components.

---

## Prior work and credits

This project directly replicates, extends, and synthesises the following:

| Work | Authors | Link |
|------|---------|------|
| **TransformerLens** (library used throughout) | Nanda, N. | [GitHub](https://github.com/TransformerLensOrg/TransformerLens) |
| **In-context Learning and Induction Heads** | Olsson et al. (2022) | [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) |
| **Interpretability in the Wild (IOI)** | Wang, Variengien, Conmy, Shlegeris, Steinhardt (2022) | [arXiv:2211.00593](https://arxiv.org/abs/2211.00593) |
| **The Logit Lens** | nostalgebraist (2020) | [LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) |
| **A Mathematical Framework for Transformer Circuits** | Elhage et al. (2021) | [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html) |
| **Mechanistic Interpretability for AI Safety -- A Review** | Bereska & Gavves (2024) | [arXiv:2404.14082](https://arxiv.org/abs/2404.14082) |
| **Progress Measures for Grokking via Mech Interp** | Nanda, Chan, Lieberum, Smith, Steinhardt (2023) | [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) |
| **Locating and Editing Factual Associations in GPT** | Meng, Bau, Andonian, Belinkov (2022) | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) |

---

## Project structure

```
circuitbreaker/
|-- README.md
|-- app.py               <- Streamlit interactive app (3 modes)
|-- notebook.ipynb       <- research notebook with full paper explanations
|-- requirements.txt
|-- figures/             <- auto-generated by notebook
    |-- attention_patterns.png
    |-- induction_scores.png
    |-- induction_loss.png
    |-- logit_lens.png
    |-- activation_patching.png
    |-- direct_logit_attribution.png
    |-- patching_vs_dla.png
```

---

## Setup

```bash
git clone https://github.com/agentjakey/mechinterp-explore
cd mechinterp-explore
pip install -r requirements.txt

# Run the interactive app
streamlit run app.py

# Or open the research notebook
jupyter notebook notebook.ipynb
```

Runs on CPU. GPT-2 Small (124M parameters) fits in 4GB RAM, no paid compute required.

**First run only:** TransformerLens downloads GPT-2 Small weights (~500MB) from HuggingFace and caches them locally. To pre-download before running the app:

```bash
python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gpt2')"
```

Note: OpenAI's original blog post cited 117M parameters for GPT-2 Small, but this was later corrected to 124M. See https://github.com/openai/gpt-2/issues/209. TransformerLens reports 163M when loaded due to its internal weight representation; the named parameter count remains 124M.

---

## Requirements

```
transformer_lens>=1.19.0
torch>=2.0.0
einops>=0.7.0
matplotlib>=3.7.0
numpy>=1.24.0
streamlit>=1.32.0
```

Python 3.10+.

---

## Key concepts

**Residual stream**: the running sum of all layer outputs, each layer reads from it and writes back to it. The logit lens reads this stream at every layer by applying the final LayerNorm and unembedding matrix to the intermediate residual state.

**Induction heads**: attention heads that implement: given repeated sequence [A][B]...[A], predict [B]. At the second occurrence of A, the head attends to the position immediately *after* the first occurrence (i.e., to B) and copies that token. Detected via the diagonal stripe at offset (1 - SEQ_LEN) on random repeated sequences. Primary mechanism behind in-context learning (Olsson et al., 2022).

**Activation patching**: replacing activations from a corrupted forward pass with those from a clean forward pass to causally identify which components drive a behavior. `hook_z` in TransformerLens is the per-head value *before* the W_O projection, patching it is equivalent to patching the full head contribution to the residual stream. Patching identifies *necessary*, not sufficient, components.

**Direct logit attribution (DLA)**: decomposing the final logit for a target token into additive contributions per head and MLP layer. Each head's contribution is `(z @ W_O) · W_U[:, target]`. DLA only measures direct projection onto the unembedding direction, it misses heads operating via indirect paths. Disagreement between DLA and patching scores is therefore informative, rather than a failure of either method.

---

## Limitations

- GPT-2 Small (124M params, 2019) is far simpler than modern LLMs. Circuits may not transfer to larger or more capable models.
- The IOI task is a controlled synthetic benchmark using a specific sentence template. Real alignment-relevant behaviors require more complex setups.
- Activation patching identifies *necessary*, not sufficient, components. A head can be causally involved without being the primary driver.
- DLA assumes additive independence of residual stream contributions, ignoring nonlinear interactions between components.
- Pearson r = 0.64 between patching and DLA reflects genuine partial disagreement. Heads with high patching scores but near-zero DLA operate via indirect causal paths that DLA cannot capture.
- Results depend on specific prompts and templates. The IOI circuit in Wang et al. (2022) is optimised for their specific template distribution.

---

## License

MIT. GPT-2 weights are subject to OpenAI's model card terms.

---

## Author

Jacob O. | UC Berkeley MIDS (incoming Fall 2026) | GitHub: [agentjakey](https://github.com/agentjakey) | Linkedin: https://www.linkedin.com/in/jacob-ortiz-ab6421348/