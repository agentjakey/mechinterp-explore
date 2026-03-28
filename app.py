"""
CircuitBreaker -- Mechanistic Interpretability Explorer
Run with: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import einops
from transformer_lens import HookedTransformer
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CircuitBreaker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# CSS template -- smooth, rounded, white + black accents
# ------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #111;
    background-color: #fff;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #111;
    letter-spacing: -0.5px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #fafafa;
    border-right: 1px solid #efefef;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #444;
}

[data-testid="stSidebar"] .stRadio label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #333;
}

hr {
    border: none;
    border-top: 1px solid #efefef;
    margin: 1.5rem 0;
}

/* Metric cards -- rounded, soft shadow */
.metric-box {
    border: 1px solid #ebebeb;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin: 0.3rem 0;
    background: #fff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease;
}

.metric-box:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}

.metric-box .label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
    margin-bottom: 0.35rem;
}

.metric-box .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #111;
}

.metric-box .sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: #bbb;
    margin-top: 0.25rem;
}

/* Info box -- pill-style left accent */
.info-box {
    border-left: 3px solid #111;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1.1rem;
    margin: 1rem 0;
    background: #f7f7f7;
    font-size: 0.82rem;
    color: #555;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    color: #999;
}

.stTabs [aria-selected="true"] {
    color: #111 !important;
    border-bottom: 2px solid #111 !important;
}

/* Buttons -- rounded pill */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    background: #111;
    color: #fff;
    border: none;
    border-radius: 999px;
    padding: 0.5rem 1.8rem;
    letter-spacing: 0.2px;
    transition: background 0.2s ease, box-shadow 0.2s ease;
}

.stButton > button:hover {
    background: #333;
    box-shadow: 0 4px 12px rgba(0,0,0,0.18);
    color: #fff;
}

/* Inputs -- softly rounded */
.stTextInput input, .stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #111 !important;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background: #fff !important;
    padding: 0.5rem 0.8rem;
    transition: border-color 0.2s ease;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #111;
    background: #fff !important;
    color: #111 !important;
}

.stTextInput label, .stTextArea label {
    color: #888 !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
}

.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #ccc !important;
}

/* Slider label */
.stSlider label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #888;
}

/* Caption */
.stCaption, caption {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #bbb;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading GPT-2 Small...")
def load_model():
    model = HookedTransformer.from_pretrained(
        "gpt2",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model.eval()
    return model


# ------------------------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------------------------
def get_top_predictions(model, prompt, k=10):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, k)
    results = [
        {"token": model.to_single_str_token(i.item()), "prob": p.item()}
        for i, p in zip(top.indices, top.values)
    ]
    return results, cache, logits


def logit_lens_analysis(model, cache, target_token):
    try:
        target_id = model.to_single_token(target_token)
    except Exception:
        return None, None, None, None

    resid_stack, labels = cache.accumulated_resid(
        layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
    )
    resid_stack = model.ln_final(resid_stack)
    logit_stack = model.unembed(resid_stack)
    prob_stack  = torch.softmax(logit_stack, dim=-1)

    correct_probs = prob_stack[:, 0, target_id].detach().cpu().numpy()
    top1_probs    = prob_stack[:, 0].max(dim=-1).values.detach().cpu().numpy()
    top1_tokens   = [
        model.to_single_str_token(prob_stack[i, 0].argmax().item())
        for i in range(len(labels))
    ]
    return labels, correct_probs, top1_tokens, top1_probs


def compute_head_dla(model, cache, target_token):
    try:
        target_id = model.to_single_token(target_token)
    except Exception:
        return None, None

    W_U_dir  = model.W_U[:, target_id]
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    head_dla = torch.zeros(n_layers, n_heads)
    mlp_dla  = torch.zeros(n_layers)

    for layer in range(n_layers):
        z      = cache[f"blocks.{layer}.attn.hook_z"][0, -1]
        W_O    = model.blocks[layer].attn.W_O
        result = einops.einsum(z, W_O, "h d, h d m -> h m")
        head_dla[layer] = einops.einsum(result, W_U_dir, "h d, d -> h")
        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]
        mlp_dla[layer] = (mlp_out * W_U_dir).sum()

    return head_dla.detach().numpy(), mlp_dla.detach().numpy()


def compute_induction_scores(model, seq_len=50, batch=10):
    rand_tokens = torch.randint(1000, model.cfg.d_vocab - 1, (batch, seq_len))
    repeated    = einops.repeat(rand_tokens, "b s -> b (2 s)")
    scores      = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    def hook(pattern, hook):
        stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
        scores[hook.layer()] = einops.reduce(stripe, "b h p -> h", "mean")

    model.run_with_hooks(
        repeated,
        return_type=None,
        fwd_hooks=[(lambda name: name.endswith("pattern"), hook)],
    )
    return scores.cpu().detach().numpy()


def compare_logit_diff(model, prompt_a, prompt_b, token_a, token_b):
    try:
        id_a = model.to_single_token(token_a)
        id_b = model.to_single_token(token_b)
    except Exception:
        return {"A": float("nan"), "B": float("nan")}

    results = {}
    for key, prompt in [("A", prompt_a), ("B", prompt_b)]:
        with torch.no_grad():
            logits = model(model.to_tokens(prompt))
        results[key] = (logits[0, -1, id_a] - logits[0, -1, id_b]).item()
    return results


# ------------------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------------------
def apply_clean_style(ax):
    ax.set_facecolor("#fff")
    ax.tick_params(colors="#666", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")
    ax.xaxis.label.set_color("#444")
    ax.yaxis.label.set_color("#444")
    ax.title.set_color("#111")
    return ax


def plot_logit_lens(labels, correct_probs, top1_tokens, top1_probs, target,
                    competing=None):
    """
    target    : the token being tracked (black bar)
    competing : optional second token of interest (dark grey bar)
                all others are light grey
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor="#fff")

    # Left -- P(target) curve
    ax = apply_clean_style(axes[0])
    ax.plot(range(len(labels)), correct_probs, "o-", color="#111", lw=1.5, ms=4)
    ax.fill_between(range(len(labels)), correct_probs, alpha=0.08, color="#111")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6,
                       fontfamily="monospace")
    ax.set_ylabel("Probability", fontsize=8)
    ax.set_title(f'P("{target.strip()}") by layer', fontsize=9,
                 fontfamily="monospace")
    ax.set_ylim(-0.02, max(correct_probs.max() * 1.3, 0.05))
    ax.grid(True, alpha=0.15, color="#ccc", linewidth=0.5)

    # Right -- top-1 per layer, with three-level coloring
    ax2 = apply_clean_style(axes[1])

    def bar_color(tok):
        if tok == target:
            return "#111"          # black  -- target token
        if competing and tok == competing:
            return "#555"          # dark grey -- competing token
        return "#ddd"              # light grey -- neither

    colors = [bar_color(t) for t in top1_tokens]
    ax2.barh(range(len(labels)), top1_probs, color=colors, height=0.6)

    for i, tok in enumerate(top1_tokens):
        ax2.text(top1_probs[i] + 0.005, i, repr(tok),
                 va="center", fontsize=6, color="#444", fontfamily="monospace")

    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=6, fontfamily="monospace")
    ax2.set_xlabel("Top-1 probability", fontsize=8)

    if competing:
        title = (f"Top-1 prediction by layer  "
                 f"[black = {repr(target.strip())}  "
                 f"grey = {repr(competing.strip())}]")
    else:
        title = "Top-1 prediction by layer  [black = target]"

    ax2.set_title(title, fontsize=8, fontfamily="monospace")
    ax2.set_xlim(0, top1_probs.max() * 1.2)

    plt.tight_layout(pad=1.5)
    return fig


def plot_dla(head_dla, mlp_dla, target):
    n_layers, n_heads = head_dla.shape
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="#fff")

    vmax = max(np.abs(head_dla).max(), 1e-3)
    ax = apply_clean_style(axes[0])
    im = ax.imshow(head_dla, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("Head", fontsize=8)
    ax.set_ylabel("Layer", fontsize=8)
    ax.set_title(
        f'Head DLA  target: "{target.strip()}"  [red = boosts, blue = suppresses]',
        fontsize=8, fontfamily="monospace"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize=7)

    thresh = vmax * 0.55
    for l in range(n_layers):
        for h in range(n_heads):
            if abs(head_dla[l, h]) > thresh:
                ax.text(h, l, f"{head_dla[l,h]:+.1f}",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold", fontfamily="monospace")

    ax2 = apply_clean_style(axes[1])
    colors = ["#222" if v > 0 else "#aaa" for v in mlp_dla]
    ax2.barh(range(n_layers), mlp_dla, color=colors, height=0.6)
    ax2.axvline(0, color="#999", lw=0.8)
    ax2.set_xlabel("DLA score", fontsize=8)
    ax2.set_ylabel("Layer", fontsize=8)
    ax2.set_title("MLP layer DLA  [dark = boosts, light = suppresses]",
                  fontsize=8, fontfamily="monospace")

    plt.tight_layout(pad=1.5)
    return fig


def plot_attention_grid(patterns, layer):
    fig, axes = plt.subplots(2, 6, figsize=(13, 4), facecolor="#fff")
    fig.suptitle(f"Attention patterns -- Layer {layer}",
                 fontsize=9, fontfamily="monospace", color="#111")
    for h, ax in enumerate(axes.flatten()):
        ax.set_facecolor("#fff")
        ax.imshow(patterns[h], cmap="Greys", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"H{h}", fontsize=7, color="#666", fontfamily="monospace")
        ax.axis("off")
    plt.tight_layout(pad=0.8)
    return fig


def plot_induction_heatmap(scores):
    n_layers, n_heads = scores.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#fff")

    ax = apply_clean_style(axes[0])
    im = ax.imshow(scores, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Head", fontsize=8)
    ax.set_ylabel("Layer", fontsize=8)
    ax.set_title("Induction scores  (layer x head)",
                 fontsize=9, fontfamily="monospace")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize=7)

    THRESHOLD = 0.6
    for l in range(n_layers):
        for h in range(n_heads):
            if scores[l, h] > THRESHOLD:
                ax.text(h, l, f"{scores[l,h]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold", fontfamily="monospace")

    ax2 = apply_clean_style(axes[1])
    ax2.barh(range(n_layers), scores.max(axis=1), color="#222", height=0.6)
    ax2.axvline(THRESHOLD, color="#999", linestyle="--", lw=1,
                label=f"threshold {THRESHOLD}")
    ax2.invert_yaxis()
    ax2.set_xlabel("Max induction score", fontsize=8)
    ax2.set_ylabel("Layer", fontsize=8)
    ax2.set_title("Peak score per layer", fontsize=9, fontfamily="monospace")
    ax2.legend(fontsize=7, framealpha=0)

    plt.tight_layout(pad=1.5)
    return fig


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:1rem;font-weight:700;"
        "color:#111;margin-bottom:0.1rem;'>CircuitBreaker</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-family:Inter,sans-serif;font-size:0.72rem;color:#bbb;'>"
        "GPT-2 Small &middot; 124M &middot; 12L &middot; 12H</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    mode = st.radio(
        "Mode",
        ["Prompt Explorer", "Contrast Pairs", "Circuit Map"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<span style='font-family:Inter,sans-serif;font-size:0.72rem;color:#ccc;'>"
        "Built on TransformerLens.<br>See notebook.ipynb for the full analysis."
        "</span>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------------------
model = load_model()

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.markdown(
    "<h1 style='font-family:Inter,sans-serif;font-size:2rem;font-weight:700;"
    "letter-spacing:-1px;margin-bottom:0;'>CircuitBreaker</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-family:Inter,sans-serif;font-size:0.9rem;color:#bbb;"
    "margin-top:0.2rem;margin-bottom:2rem;font-weight:400;'>"
    "Mechanistic interpretability of GPT-2 Small live in your browser.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ==============================================================================
# MODE 1: PROMPT EXPLORER
# ==============================================================================
if mode == "Prompt Explorer":

    col_left, col_right = st.columns([3, 1])
    with col_left:
        prompt = st.text_area(
            "Prompt",
            value="The Eiffel Tower is located in the city of",
            height=80,
        )
    with col_right:
        target_token = st.text_input("Target token", value=" Paris")
        k_preds = st.slider("Top-k", 3, 20, 8)

    st.markdown(" ")
    run = st.button("Run", use_container_width=False)

    if run and prompt.strip():
        with st.spinner("Running forward pass..."):
            top_preds, cache, logits = get_top_predictions(model, prompt, k=k_preds)

        st.markdown("**Top predictions**")
        pred_cols = st.columns(min(k_preds, 8))
        for i, pred in enumerate(top_preds[:8]):
            with pred_cols[i]:
                is_target = pred["token"].strip() == target_token.strip()
                border = "border: 2px solid #111;" if is_target else ""
                st.markdown(
                    f'<div class="metric-box" style="{border}">'
                    f'<div class="label">#{i+1}</div>'
                    f'<div class="value" style="font-size:1rem;">{repr(pred["token"])}</div>'
                    f'<div class="sub">{pred["prob"]:.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(" ")

        target_valid = True
        try:
            model.to_single_token(target_token)
        except Exception:
            st.warning(
                f"{repr(target_token)} is not a single token. "
                "Logit lens and DLA will be skipped."
            )
            target_valid = False

        tabs = st.tabs(["Logit Lens", "Direct Logit Attribution", "Attention Patterns"])

        # -- Logit lens (no competing token in single-prompt mode)
        with tabs[0]:
            if target_valid:
                with st.spinner("Computing logit lens..."):
                    labels, correct_probs, top1_tokens, top1_probs = \
                        logit_lens_analysis(model, cache, target_token)

                if labels is not None:
                    fig = plot_logit_lens(
                        labels, correct_probs, top1_tokens, top1_probs, target_token
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    peak_layer = int(np.argmax(correct_probs))
                    peak_prob  = correct_probs[peak_layer]
                    never_top1 = target_token not in top1_tokens

                    if never_top1:
                        note = (
                            f"{repr(target_token)} never reaches top-1 at any layer. "
                            f"Peak P = {peak_prob:.3f} at {labels[peak_layer]}. "
                            "This is a GPT-2 Small model limitation, not a code error."
                        )
                    else:
                        first = next(
                            i for i, t in enumerate(top1_tokens) if t == target_token
                        )
                        note = (
                            f"{repr(target_token)} first reaches top-1 at "
                            f"{labels[first]} (P = {top1_probs[first]:.3f}). "
                            f"Final P = {correct_probs[-1]:.3f}."
                        )
                    st.markdown(
                        f'<div class="info-box">{note}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Enter a valid single token to enable logit lens.")

        # -- DLA
        with tabs[1]:
            if target_valid:
                with st.spinner("Computing direct logit attribution..."):
                    head_dla, mlp_dla = compute_head_dla(model, cache, target_token)

                if head_dla is not None:
                    fig = plot_dla(head_dla, mlp_dla, target_token)
                    st.pyplot(fig)
                    plt.close(fig)

                    flat = head_dla.flatten()
                    bc, sc = st.columns(2)
                    with bc:
                        st.markdown("**Top boosting heads**")
                        for idx in np.argsort(flat)[-3:][::-1]:
                            l, h = divmod(idx, model.cfg.n_heads)
                            st.markdown(
                                f'<div class="metric-box">'
                                f'<div class="label">L{l} H{h}</div>'
                                f'<div class="value" style="font-size:1.1rem;">'
                                f'+{flat[idx]:.2f}</div></div>',
                                unsafe_allow_html=True,
                            )
                    with sc:
                        st.markdown("**Top suppressing heads**")
                        for idx in np.argsort(flat)[:3]:
                            l, h = divmod(idx, model.cfg.n_heads)
                            st.markdown(
                                f'<div class="metric-box">'
                                f'<div class="label">L{l} H{h}</div>'
                                f'<div class="value" style="font-size:1.1rem;">'
                                f'{flat[idx]:.2f}</div></div>',
                                unsafe_allow_html=True,
                            )
            else:
                st.info("Enter a valid single token to enable DLA.")

        # -- Attention patterns
        with tabs[2]:
            layer_sel = st.slider("Layer", 0, model.cfg.n_layers - 1, 5)
            patterns  = cache["pattern", layer_sel][0].cpu().numpy()
            fig = plot_attention_grid(patterns, layer_sel)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"Attention matrices for all 12 heads at layer {layer_sel}. "
                "Dark = high attention weight."
            )


# ==============================================================================
# MODE 2: CONTRAST PAIRS
# ==============================================================================
elif mode == "Contrast Pairs":

    st.markdown("### Contrast Pairs")
    st.markdown(
        '<div class="info-box">'
        "Enter two prompts that differ by one element. "
        "CircuitBreaker computes the logit difference between two tokens "
        "and shows side-by-side logit lens traces. "
        "In the bar charts: black = Token A, dark grey = Token B, light = neither."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(" ")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Prompt A**")
        prompt_a = st.text_area(
            "Prompt A",
            value="When John and Mary went to the store, John gave a bottle of milk to",
            height=90,
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("**Prompt B**")
        prompt_b = st.text_area(
            "Prompt B",
            value="When John and Mary went to the store, Mary gave a bottle of milk to",
            height=90,
            label_visibility="collapsed",
        )

    t1, t2 = st.columns(2)
    with t1:
        token_a = st.text_input("Token A (expected)", value=" Mary")
    with t2:
        token_b = st.text_input("Token B (competing)", value=" John")

    st.markdown(" ")
    if st.button("Compare", use_container_width=False):
        with st.spinner("Running both prompts..."):
            diffs    = compare_logit_diff(model, prompt_a, prompt_b, token_a, token_b)
            _, ca, _ = get_top_predictions(model, prompt_a, k=5)
            _, cb, _ = get_top_predictions(model, prompt_b, k=5)

        st.markdown(" ")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="label">Logit diff -- Prompt A</div>'
                f'<div class="value">{diffs["A"]:+.3f}</div>'
                f'<div class="sub">positive = favours {repr(token_a)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="label">Logit diff -- Prompt B</div>'
                f'<div class="value">{diffs["B"]:+.3f}</div>'
                f'<div class="sub">negative = favours {repr(token_b)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m3:
            swing = diffs["A"] - diffs["B"]
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="label">Swing (A minus B)</div>'
                f'<div class="value">{swing:+.3f}</div>'
                f'<div class="sub">total shift from the edit</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(" ")
        st.markdown("**Logit lens -- side by side**")
        lc1, lc2 = st.columns(2)

        for col, cache_, label in [
            (lc1, ca, "Prompt A"),
            (lc2, cb, "Prompt B"),
        ]:
            labels, correct_probs, top1_tokens, top1_probs = \
                logit_lens_analysis(model, cache_, token_a)
            if labels is not None:
                with col:
                    st.markdown(f"*{label}*")
                    # Pass token_b as competing so its bars show as dark grey
                    fig = plot_logit_lens(
                        labels, correct_probs, top1_tokens, top1_probs,
                        target=token_a, competing=token_b
                    )
                    st.pyplot(fig)
                    plt.close(fig)


# ==============================================================================
# MODE 3: CIRCUIT MAP
# ==============================================================================
elif mode == "Circuit Map":

    st.markdown("### Circuit Map")
    st.markdown(
        '<div class="info-box">'
        "Computes induction scores across all 144 (layer, head) combinations "
        "using random repeated sequences. High scores identify genuine induction heads: "
        "the primary substrate of in-context learning (Olsson et al., 2022)."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(" ")

    p1, p2 = st.columns(2)
    with p1:
        seq_len = st.slider("Sequence length", 20, 100, 50)
    with p2:
        batch = st.slider("Batch size", 5, 30, 10)

    st.markdown(" ")
    if st.button("Compute", use_container_width=False):
        with st.spinner("Running induction detection..."):
            scores = compute_induction_scores(model, seq_len=seq_len, batch=batch)

        fig = plot_induction_heatmap(scores)
        st.pyplot(fig)
        plt.close(fig)

        THRESHOLD = 0.6
        strong = [
            (l, h, scores[l, h])
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            if scores[l, h] > THRESHOLD
        ]
        strong.sort(key=lambda x: -x[2])

        if strong:
            st.markdown(f"**Strong induction heads  (score > {THRESHOLD})**")
            cols = st.columns(min(len(strong), 8))
            for col, (l, h, sc) in zip(cols, strong):
                with col:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="label">L{l} H{h}</div>'
                        f'<div class="value" style="font-size:1.1rem;">{sc:.3f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown(" ")
            st.markdown(
                '<div class="info-box">'
                f"Found {len(strong)} strong induction heads. "
                "These heads implement the [A][B]...[A] -> [B] copying algorithm "
                "and are the primary mechanism behind in-context learning in GPT-2 Small."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No heads exceeded the threshold. Try a longer sequence length.")