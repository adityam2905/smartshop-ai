"""
Phase 5: Streamlit UI & Online Learning Loop
The main application entry point.

Run with:
    streamlit run app.py

Environment variables (optional):
    SERPAPI_KEY   — your SerpAPI key for live Google Shopping results
                    (falls back to rich mock data if not set)
"""

import os
import numpy as np
import streamlit as st

# ── Page config — MUST be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title  = "🛍️ AI Deal Hunter",
    page_icon   = "🛍️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Local module imports ───────────────────────────────────────────────────────
from scraper import (
    search_products,
    features_to_obs,
    update_user_preference,
    get_user_preference,
)
from train_agent import load_agent, fine_tune_on_feedback

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH          = "dqn_shopping_agent.zip"
FEEDBACK_FINETUNE_EVERY  = 3      # fine-tune after every N feedback signals
LIKE_REWARD         = +20.0
DISLIKE_REWARD      = -20.0
LIKE_PREF_DELTA     = +0.05       # how much a Like shifts category preference
DISLIKE_PREF_DELTA  = -0.05
SCAM_THRESHOLD      = 0.30
FINETUNE_GRAD_STEPS = 50
SEARCH_RESULT_LIMIT  = 12


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Global ─────────────────────────────────────────── */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #f0f0f0;
        }
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.04);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* ── Hero header ─────────────────────────────────────── */
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .hero-sub {
            text-align: center;
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        /* ── Product card ────────────────────────────────────── */
        .product-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            transition: border-color 0.2s;
        }
        .product-card:hover {
            border-color: rgba(167,139,250,0.5);
        }
        .product-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 0.5rem;
        }
        .price-row {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.7rem;
        }
        .price-current {
            font-size: 1.5rem;
            font-weight: 800;
            color: #34d399;
        }
        .price-was {
            font-size: 0.9rem;
            color: #64748b;
            text-decoration: line-through;
        }
        .discount-badge {
            background: #ef4444;
            color: white;
            padding: 2px 8px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        /* ── Trust / confidence bars ─────────────────────────── */
        .meta-row {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 0.8rem;
            font-size: 0.82rem;
            color: #94a3b8;
        }
        .meta-chip {
            background: rgba(255,255,255,0.07);
            border-radius: 8px;
            padding: 2px 10px;
        }
        .trust-high  { color: #34d399; }
        .trust-med   { color: #fbbf24; }
        .trust-low   { color: #f87171; }

        /* ── Feedback buttons ────────────────────────────────── */
        div[data-testid="column"] button {
            border-radius: 50px !important;
            font-weight: 600 !important;
            width: 100% !important;
        }

        /* ── Stats bar at the top ────────────────────────────── */
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 2rem;
            background: rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 0.8rem 1.5rem;
            margin-bottom: 1.5rem;
            font-size: 0.88rem;
        }
        .stat-item { text-align: center; }
        .stat-val  { font-size: 1.4rem; font-weight: 800; color: #a78bfa; }
        .stat-lbl  { color: #64748b; font-size: 0.75rem; }

        /* ── Sidebar metric cards ────────────────────────────── */
        .pref-bar-wrap { margin-bottom: 0.4rem; }
        .pref-bar-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: #94a3b8;
            margin-bottom: 2px;
        }

        /* ── Empty state ─────────────────────────────────────── */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: #64748b;
        }
        .empty-icon { font-size: 4rem; margin-bottom: 1rem; }

        /* ── Scam warning card ───────────────────────────────── */
        .scam-card {
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.3);
            border-radius: 12px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 0.8rem;
            font-size: 0.85rem;
            color: #fca5a5;
        }

        /* ── Toast-like feedback confirm ─────────────────────── */
        .feedback-toast {
            background: rgba(52,211,153,0.12);
            border: 1px solid rgba(52,211,153,0.35);
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-size: 0.82rem;
            color: #6ee7b7;
            margin-top: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "model":              None,       # loaded DQN
        "model_loaded":       False,
        "search_query":       "",
        "search_results":     [],         # list of feature dicts from scraper
        "recommendations":    [],         # subset where agent chose Action 1
        "skipped_count":      0,
        "scams_caught_last_search": 0,
        "feedback_buffer":    [],         # list of experience dicts for fine-tuning
        "feedback_counts":    {"like": 0, "dislike": 0},
        "scams_caught":       0,          # items agent filtered with trust < 0.3
        "total_searches":     0,
        "last_query":         "",
        "fine_tune_count":    0,          # how many times we've fine-tuned
        "agent_confidence":   {},         # item_index → q-value spread (optional display)
        "toast":              {},         # {item_index: "like"/"dislike"} for UI feedback
        "use_mock":           not bool(os.environ.get("SERPAPI_KEY", "")),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so it only loads once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    """Load the DQN model once and cache it for the Streamlit session."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found. Run `python train_agent.py` first."
    try:
        model = load_agent(MODEL_PATH)
        return model, None
    except Exception as exc:
        return None, str(exc)


def get_preference_snapshot() -> tuple[tuple[str, float], ...]:
    """Stable cache key input for search results that depend on preferences."""
    categories = [
        "Electronics", "Clothing", "Home & Garden",
        "Sports", "Books", "Toys", "Beauty", "Automotive",
    ]
    return tuple((category, get_user_preference(category)) for category in categories)


@st.cache_data(show_spinner=False, ttl=300)
def cached_search(
    query: str,
    use_mock: bool,
    num_results: int,
    _pref_snapshot: tuple[tuple[str, float], ...],
) -> list[dict]:
    # The snapshot is part of the cache key so feedback changes invalidate stale results.
    return search_products(query, use_mock=use_mock, num_results=num_results)


# ─────────────────────────────────────────────────────────────────────────────
# Core: run agent inference over a list of feature dicts
# ─────────────────────────────────────────────────────────────────────────────
def run_agent_inference(model, feature_list: list[dict]) -> tuple[list[dict], list[dict], int]:
    """
    Passes every product through the DQN model.
    Returns:
        recommendations  — items where action == 1 (Recommend)
        skipped          — items where action == 0 (Skip)
        scams_caught     — count of items filtered due to trust < SCAM_THRESHOLD
    """
    recommendations = []
    skipped         = []
    scams_caught    = 0

    for i, feat in enumerate(feature_list):
        obs    = features_to_obs(feat).reshape(1, -1)
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        # Annotate the feature dict with agent metadata
        feat["_action"]     = action
        feat["_item_index"] = i

        # Hard override: never recommend a known scam (trust < threshold)
        is_scam_domain = feat["site_trust_score"] < SCAM_THRESHOLD
        if is_scam_domain:
            scams_caught += 1
            feat["_scam_flag"] = True
            skipped.append(feat)
            continue

        feat["_scam_flag"] = False

        if action == 1:
            recommendations.append(feat)
        else:
            skipped.append(feat)

    return recommendations, skipped, scams_caught


# ─────────────────────────────────────────────────────────────────────────────
# Feedback handler
# ─────────────────────────────────────────────────────────────────────────────
def handle_feedback(feat: dict, sentiment: str) -> None:
    """
    Called when user clicks 👍 or 👎 on a product card.
    1. Records the experience in the feedback buffer.
    2. Updates category preference score.
    3. Triggers fine-tuning every FEEDBACK_FINETUNE_EVERY signals.
    """
    reward      = LIKE_REWARD if sentiment == "like" else DISLIKE_REWARD
    pref_delta  = LIKE_PREF_DELTA if sentiment == "like" else DISLIKE_PREF_DELTA

    # Build the experience tuple
    obs      = features_to_obs(feat)
    next_obs = np.zeros(4, dtype=np.float32)   # terminal-style next state

    experience = {
        "obs":      obs,
        "action":   1,           # the agent recommended it (that's why we're rating it)
        "reward":   reward,
        "next_obs": next_obs,
        "done":     True,
    }

    st.session_state.feedback_buffer.append(experience)
    st.session_state.feedback_counts[sentiment] += 1

    # Update per-category preference
    update_user_preference(feat.get("category", "General"), pref_delta)

    # Mark toast for this item
    st.session_state.toast[feat["_item_index"]] = sentiment

    # Trigger fine-tuning if we've accumulated enough feedback
    total_feedback = (
        st.session_state.feedback_counts["like"] +
        st.session_state.feedback_counts["dislike"]
    )
    if total_feedback % FEEDBACK_FINETUNE_EVERY == 0 and st.session_state.model:
        with st.spinner("🧠 Agent is learning from your feedback…"):
            st.session_state.model = fine_tune_on_feedback(
                st.session_state.model,
                st.session_state.feedback_buffer[-FEEDBACK_FINETUNE_EVERY:],
                gradient_steps=FINETUNE_GRAD_STEPS,
            )
            st.session_state.fine_tune_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def trust_class(score: float) -> str:
    if score >= 0.75: return "trust-high"
    if score >= 0.45: return "trust-med"
    return "trust-low"

def trust_label(score: float) -> str:
    if score >= 0.75: return "✅ Trusted"
    if score >= 0.45: return "⚠️ Moderate"
    return "🚨 Suspicious"

def discount_label(disc: float) -> str:
    pct = int(disc * 100)
    return f"-{pct}%" if pct > 0 else ""

def format_price(p: float) -> str:
    return f"${p:,.2f}"


def render_product_card(feat: dict, idx: int) -> None:
    """Renders a single product card with feedback buttons."""
    item_i   = feat["_item_index"]
    toast    = st.session_state.toast.get(item_i)

    disc_pct = feat["discount_percentage"]
    trust    = feat["site_trust_score"]
    price    = feat["price"]
    mkt      = feat.get("market_avg", 0)
    disc_lbl = discount_label(disc_pct)

    st.markdown('<div class="product-card">', unsafe_allow_html=True)

    # ── Title ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="product-title">{feat["product_name"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Price row ──────────────────────────────────────────────────────────
    was_html = (
        f'<span class="price-was">was {format_price(mkt)}</span>'
        if mkt and mkt > price * 1.05
        else ""
    )
    badge_html = (
        f'<span class="discount-badge">{disc_lbl}</span>'
        if disc_lbl else ""
    )
    st.markdown(
        f'<div class="price-row">'
        f'  <span class="price-current">{format_price(price)}</span>'
        f'  {was_html}{badge_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Meta chips ─────────────────────────────────────────────────────────
    t_cls = trust_class(trust)
    t_lbl = trust_label(trust)
    cat   = feat.get("category", "General")
    src   = feat.get("source", "Unknown")
    pref  = feat.get("user_preference_score", 0.5)

    st.markdown(
        f'<div class="meta-row">'
        f'  <span class="meta-chip"><span class="{t_cls}">{t_lbl}</span> &nbsp;'
        f'trust {trust:.0%}</span>'
        f'  <span class="meta-chip">🏷️ {cat}</span>'
        f'  <span class="meta-chip">🏪 {src}</span>'
        f'  <span class="meta-chip">⭐ Pref {pref:.0%}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Link ───────────────────────────────────────────────────────────────
    url = feat.get("site_url", "")
    if url:
        st.markdown(
            f'<a href="{url}" target="_blank" style="font-size:0.8rem;color:#60a5fa;">'
            f'🔗 View on {src}</a>',
            unsafe_allow_html=True,
        )

    # ── Feedback buttons ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    if toast == "like":
        st.markdown(
            '<div class="feedback-toast">👍 Thanks! Teaching the agent you liked this.</div>',
            unsafe_allow_html=True,
        )
    elif toast == "dislike":
        st.markdown(
            '<div class="feedback-toast" style="background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.3);color:#fca5a5;">'
            '👎 Noted! Agent will learn to skip similar items.</div>',
            unsafe_allow_html=True,
        )
    else:
        col_like, col_dis, col_gap = st.columns([1, 1, 2])
        with col_like:
            if st.button("👍 Like", key=f"like_{item_i}_{idx}"):
                handle_feedback(feat, "like")
                st.rerun()
        with col_dis:
            if st.button("👎 Dislike", key=f"dis_{item_i}_{idx}"):
                handle_feedback(feat, "dislike")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(model_loaded: bool) -> None:
    """Renders the left sidebar with session stats and settings."""
    with st.sidebar:
        st.markdown("## 🤖 Agent Status")

        if model_loaded:
            st.success("DQN model loaded", icon="✅")
        else:
            st.error("Model not found", icon="❌")
            st.caption(f"Run `python train_agent.py` to generate `{MODEL_PATH}`")

        st.divider()

        # ── Session stats ──────────────────────────────────────────────────
        st.markdown("## 📊 Session Stats")
        likes    = st.session_state.feedback_counts["like"]
        dislikes = st.session_state.feedback_counts["dislike"]
        scams    = st.session_state.scams_caught
        searches = st.session_state.total_searches
        ft_count = st.session_state.fine_tune_count

        col1, col2 = st.columns(2)
        col1.metric("Searches",     searches)
        col2.metric("Scams Caught", scams)
        col1.metric("👍 Likes",     likes)
        col2.metric("👎 Dislikes",  dislikes)
        st.metric("🧠 Fine-tunes",  ft_count,
                  help=f"Agent has retrained {ft_count} time(s) on your feedback.")

        # Progress to next fine-tune
        total_fb = likes + dislikes
        next_ft  = FEEDBACK_FINETUNE_EVERY - (total_fb % FEEDBACK_FINETUNE_EVERY)
        if total_fb > 0 and next_ft != FEEDBACK_FINETUNE_EVERY:
            st.caption(f"Next fine-tune in **{next_ft}** more feedback signal(s).")
            st.progress((FEEDBACK_FINETUNE_EVERY - next_ft) / FEEDBACK_FINETUNE_EVERY)

        st.divider()

        # ── Category preferences ───────────────────────────────────────────
        st.markdown("## 🎯 Learned Preferences")
        categories = [
            "Electronics", "Clothing", "Home & Garden",
            "Sports", "Books", "Toys", "Beauty", "Automotive",
        ]
        for cat in categories:
            score = get_user_preference(cat)
            st.markdown(
                f'<div class="pref-bar-wrap">'
                f'  <div class="pref-bar-label"><span>{cat}</span>'
                f'  <span>{score:.0%}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(score)

        st.divider()

        # ── Settings ───────────────────────────────────────────────────────
        st.markdown("## ⚙️ Settings")
        serpapi_key = os.environ.get("SERPAPI_KEY", "")
        if serpapi_key:
            st.success("SerpAPI key detected — live search active", icon="🌐")
            st.session_state.use_mock = False
        else:
            st.info("No SERPAPI_KEY set — using mock data", icon="📦")
            st.session_state.use_mock = True

        st.session_state.use_mock = st.checkbox(
            "Force mock data", value=st.session_state.use_mock
        )

        st.caption(
            "Set `SERPAPI_KEY` env var for live Google Shopping results. "
            "Mock data includes realistic scam + legit listings for demo purposes."
        )

        st.divider()
        st.markdown(
            "<div style='font-size:0.75rem;color:#475569;text-align:center;'>"
            "E-Commerce Deal Hunter · DQN Agent<br>"
            "Built with Stable Baselines3 + Streamlit"
            "</div>",
            unsafe_allow_html=True,
        )


def render_stats_bar() -> None:
    """Compact stats bar shown above search results."""
    recs   = len(st.session_state.recommendations)
    skips  = st.session_state.skipped_count
    scams  = st.session_state.scams_caught_last_search
    total  = len(st.session_state.search_results)

    if total == 0:
        return

    blocked = min(scams, skips)
    low_value_skips = max(skips - blocked, 0)

    st.markdown(
        f'<div class="stats-bar">'
        f'  <div class="stat-item"><div class="stat-val">{total}</div>'
        f'      <div class="stat-lbl">Products Found</div></div>'
        f'  <div class="stat-item"><div class="stat-val" style="color:#34d399">{recs}</div>'
        f'      <div class="stat-lbl">Recommended</div></div>'
        f'  <div class="stat-item"><div class="stat-val" style="color:#f87171">{scams}</div>'
        f'      <div class="stat-lbl">Scams Blocked</div></div>'
        f'  <div class="stat-item"><div class="stat-val" style="color:#94a3b8">{low_value_skips}</div>'
        f'      <div class="stat-lbl">Skipped (Low value)</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    init_session_state()

    # ── Load model ─────────────────────────────────────────────────────────
    model, model_err = get_model()
    if model and not st.session_state.model_loaded:
        st.session_state.model        = model
        st.session_state.model_loaded = True

    # ── Sidebar ────────────────────────────────────────────────────────────
    render_sidebar(model_loaded=st.session_state.model_loaded)

    # ── Hero header ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-title">🛍️ AI Deal Hunter</div>'
        '<div class="hero-sub">'
        'A reinforcement-learned agent that finds real deals, blocks scams, '
        'and personalises to <em>your</em> taste in real time.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Model error banner ─────────────────────────────────────────────────
    if model_err:
        st.error(
            f"⚠️ **Model not loaded:** {model_err}\n\n"
            "Run the following commands first:\n"
            "```bash\n"
            "python data_generator.py\n"
            "python train_agent.py --timesteps 100000\n"
            "```",
            icon="🤖",
        )

    # ── Search bar ─────────────────────────────────────────────────────────
    with st.form("search_form", clear_on_submit=False):
        search_col, btn_col = st.columns([5, 1])
        with search_col:
            query = st.text_input(
                label       = "Search",
                placeholder = "e.g. Sony Headphones, Nike Shoes, iPhone 15 …",
                label_visibility = "collapsed",
                key         = "search_query",
            )
        with btn_col:
            search_clicked = st.form_submit_button("🔍 Search", use_container_width=True, type="primary")

    # ── Quick search chips ─────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;margin:-0.5rem 0 1.5rem;'>"
        "<span style='font-size:0.8rem;color:#64748b;'>Try: </span>",
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(5)
    suggestions = ["Sony Headphones", "Nike Shoes", "iPhone 15", "MacBook Pro", "Gaming Chair"]
    chip_query  = None
    for i, sug in enumerate(suggestions):
        with chip_cols[i]:
            if st.button(sug, key=f"chip_{i}", use_container_width=True):
                chip_query = sug
                st.session_state.search_query = sug

    # ── Resolve final query ────────────────────────────────────────────────
    final_query = chip_query or (query if search_clicked else "")

    # ── Execute search ─────────────────────────────────────────────────────
    if final_query and final_query != st.session_state.last_query:
        st.session_state.last_query = final_query
        st.session_state.toast      = {}    # clear old feedback toasts

        if not st.session_state.model_loaded:
            st.warning("Train the model first before searching.", icon="⚠️")
        else:
            with st.spinner(f'Searching for **"{final_query}"** and running AI analysis…'):
                # 1. Fetch products
                feature_list = cached_search(
                    final_query,
                    st.session_state.use_mock,
                    SEARCH_RESULT_LIMIT,
                    get_preference_snapshot(),
                )
                # 2. Run agent inference
                recs, skipped, scams_caught = run_agent_inference(
                    st.session_state.model, feature_list
                )
                # 3. Store in session state
                st.session_state.search_results  = feature_list
                st.session_state.recommendations = recs
                st.session_state.skipped_count   = len(skipped)
                st.session_state.scams_caught_last_search = scams_caught
                st.session_state.scams_caught    += scams_caught
                st.session_state.total_searches  += 1

    # ── Results area ───────────────────────────────────────────────────────
    if st.session_state.recommendations:
        render_stats_bar()

        recs = st.session_state.recommendations

        # Sort by discount descending so the best deals appear first
        recs_sorted = sorted(recs, key=lambda x: x["discount_percentage"], reverse=True)

        # Two-column card grid
        left_col, right_col = st.columns(2)
        for i, feat in enumerate(recs_sorted):
            target_col = left_col if i % 2 == 0 else right_col
            with target_col:
                render_product_card(feat, idx=i)

    elif st.session_state.search_results:
        # Agent filtered everything
        scams   = st.session_state.scams_caught
        skipped = st.session_state.skipped_count
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">🤖</div>'
            '<h3 style="color:#e2e8f0;">No deals worth showing</h3>'
            '<p>The agent analysed all results and found nothing worth recommending. '
            f'{scams} suspicious listing(s) were silently blocked.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    elif not final_query:
        # Landing state
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">🔍</div>'
            '<h3 style="color:#e2e8f0;">Ready to hunt deals</h3>'
            '<p>Type a product above or click a suggestion.<br>'
            'The AI will filter out scams and surface only the best offers.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Fine-tuning notification ────────────────────────────────────────────
    if st.session_state.fine_tune_count > 0:
        total_fb = (
            st.session_state.feedback_counts["like"] +
            st.session_state.feedback_counts["dislike"]
        )
        # Show a banner after each fine-tune trigger
        if total_fb % FEEDBACK_FINETUNE_EVERY == 0 and total_fb > 0:
            st.toast(
                f"🧠 Agent retrained! ({st.session_state.fine_tune_count} fine-tune(s) so far)",
                icon="✅",
            )


if __name__ == "__main__":
    main()
