"""
TripGenie Streamlit Demo Application.

Provides an interactive UI that calls the FastAPI backend and presents
the full trip planning results in a polished, easy-to-explore interface.

Run:
    streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("TRIPGENIE_API_URL", "http://localhost:8000")
PLAN_URL = f"{API_BASE}/trip/plan"
EVAL_URL = f"{API_BASE}/eval/run"
HEALTH_URL = f"{API_BASE}/health"

SAMPLE_QUERIES = [
    "4-day Amsterdam trip for a couple, mid-range budget, close to museums, canal cruise included.",
    "Paris for 5 days, family of four with kids. Museums, Eiffel Tower, Versailles day trip.",
    "Barcelona solo 3 days, budget traveller. Architecture, food markets, nightlife.",
    "Rome honeymoon 6 days, luxury. Private tours, Colosseum, Pompeii day trip.",
    "Lisbon 4 days, two adults, mid-range. Fado, Alfama, Sintra day trip.",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TripGenie — Agentic Travel Assistant",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .hotel-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #2196F3;
    }
    .hotel-card-top {
        border-left-color: #4CAF50;
    }
    .metric-box {
        background: #e8f4fd;
        border-radius: 6px;
        padding: 0.6rem;
        text-align: center;
    }
    .score-badge {
        display: inline-block;
        background: #2196F3;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .quality-excellent { color: #4CAF50; font-weight: 600; }
    .quality-good { color: #2196F3; font-weight: 600; }
    .quality-fair { color: #FF9800; font-weight: 600; }
    .day-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.6rem;
    }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_api_health() -> dict | None:
    try:
        resp = httpx.get(HEALTH_URL, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _call_plan_api(payload: dict) -> dict | None:
    try:
        with st.spinner("TripGenie is planning your trip..."):
            resp = httpx.post(PLAN_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text[:300]}")
    except httpx.ConnectError:
        st.error(
            "Cannot connect to the TripGenie API. "
            "Start it with: `uvicorn app.main:create_app --factory --port 8000`"
        )
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    return None


def _call_eval_api(num_samples: int, verbose: bool) -> dict | None:
    try:
        with st.spinner(f"Running evaluation on {num_samples} sample queries..."):
            resp = httpx.post(
                EVAL_URL,
                json={"num_samples": num_samples, "verbose": verbose},
                timeout=120,
            )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Eval API error {resp.status_code}: {resp.text[:300]}")
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
    return None


def _score_color(score: float) -> str:
    if score >= 0.85:
        return "quality-excellent"
    elif score >= 0.70:
        return "quality-good"
    return "quality-fair"


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.55:
        return "Fair"
    return "Needs improvement"


def _price_level_str(level: int) -> str:
    return {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}.get(level, "$$$")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## TripGenie")
    st.markdown("*Agentic Travel Planning Assistant*")
    st.divider()

    # API health indicator
    health = _check_api_health()
    if health:
        st.success(f"API Online — {health.get('version', '?')} [{health.get('llm_provider', '?')}]")
    else:
        st.warning("API Offline — Start the backend first")

    st.divider()
    st.markdown("### Quick Sample Queries")
    for i, q in enumerate(SAMPLE_QUERIES):
        if st.button(f"Sample {i+1}", key=f"sample_{i}", use_container_width=True):
            st.session_state["query_input"] = q

    st.divider()
    st.markdown("### Settings")
    use_llm = st.toggle("Use LLM", value=True, help="Toggle off to use faster mock mode")
    top_k = st.slider("Max hotel recommendations", 3, 8, 5)

    st.divider()
    st.caption("Built with FastAPI + Streamlit | Mock LLM mode: no API key needed")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">TripGenie ✈</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Your agentic travel planning assistant — powered by AI</p>',
    unsafe_allow_html=True,
)

# Tab layout
tab_plan, tab_eval, tab_about = st.tabs(["Trip Planner", "Evaluation", "About"])

# ===========================================================================
# TAB 1: Trip Planner
# ===========================================================================
with tab_plan:
    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_area(
            "Describe your ideal trip",
            value=st.session_state.get("query_input", ""),
            height=120,
            placeholder=(
                "e.g. I want a 4-day Amsterdam trip for a couple, mid-range budget, "
                "close to museums and public transport, with a canal cruise and one day trip."
            ),
            key="query_area",
        )

    with col2:
        st.markdown("**Optional overrides**")
        city_override = st.selectbox(
            "City", ["", "Amsterdam", "Paris", "Barcelona", "Rome", "Lisbon"]
        )
        days_override = st.number_input("Days", 0, 30, 0, help="0 = extract from query")
        budget_override = st.selectbox("Budget", ["", "budget", "mid", "upper_mid", "luxury"])
        travelers_override = st.number_input("Travelers", 0, 20, 0)

    plan_clicked = st.button("Plan My Trip", type="primary", use_container_width=False)

    if plan_clicked and query.strip():
        payload = {
            "query": query.strip(),
            "use_llm": use_llm,
        }
        if city_override:
            payload["city"] = city_override
        if days_override > 0:
            payload["days"] = int(days_override)
        if budget_override:
            payload["budget_level"] = budget_override
        if travelers_override > 0:
            payload["travelers"] = int(travelers_override)

        result = _call_plan_api(payload)

        if result:
            st.session_state["plan_result"] = result

    elif plan_clicked and not query.strip():
        st.warning("Please enter a travel request.")

    # Display results
    if "plan_result" in st.session_state:
        r = st.session_state["plan_result"]

        st.divider()

        # --- Parsed Intent ---
        intent = r.get("parsed_intent", {})
        st.markdown("### Parsed Trip Intent")
        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        ic1.metric("City", intent.get("city", "—"))
        ic2.metric("Days", intent.get("days", "—"))
        ic3.metric("Travelers", intent.get("travelers", "—"))
        ic4.metric("Budget", intent.get("budget_level", "—").replace("_", " ").title())
        ic5.metric("Style", intent.get("travel_style", "—").title())

        if intent.get("interests"):
            st.markdown(
                "**Interests detected:** "
                + " · ".join(f"`{i}`" for i in intent["interests"])
            )
        if intent.get("special_requests"):
            st.markdown(
                "**Special requests:** "
                + " · ".join(f"`{s}`" for s in intent["special_requests"])
            )

        st.divider()

        # --- Results in two columns ---
        left, right = st.columns([1.2, 1])

        # --- Hotels ---
        with left:
            st.markdown("### Hotel Recommendations")
            ranked_hotels = r.get("ranked_hotels", [])
            for i, rh in enumerate(ranked_hotels):
                hotel = rh.get("hotel", {})
                score = rh.get("score", 0)
                card_class = "hotel-card hotel-card-top" if i == 0 else "hotel-card"
                rank_emoji = ["🥇", "🥈", "🥉", "4.", "5."][i] if i < 5 else f"{i+1}."

                with st.expander(
                    f"{rank_emoji} **{hotel.get('name', '—')}** — "
                    f"{_price_level_str(hotel.get('price_level', 2))} | "
                    f"Score: {score:.0%}",
                    expanded=(i == 0),
                ):
                    st.markdown(f"*{hotel.get('description', '')}*")
                    ft1, ft2, ft3 = st.columns(3)
                    ft1.metric("Review", f"{hotel.get('avg_review_score', 0):.1f}/10")
                    ft2.metric("Location", f"{hotel.get('location_score', 0):.1f}/10")
                    ft3.metric("Price", hotel.get("price_label", "—"))

                    # Feature badges
                    features = []
                    if hotel.get("near_museum"):
                        features.append("Near Museums")
                    if hotel.get("near_public_transport"):
                        features.append("Near Transport")
                    if hotel.get("romantic"):
                        features.append("Romantic")
                    if hotel.get("family_friendly"):
                        features.append("Family Friendly")
                    if hotel.get("near_nightlife"):
                        features.append("Near Nightlife")
                    if features:
                        st.markdown(" | ".join(f"✓ {f}" for f in features))

                    # Explanation
                    explanation = rh.get("explanation", "")
                    if explanation:
                        st.caption(f"Ranking reason: {explanation}")

                    # Feature contributions
                    contributions = rh.get("feature_contributions", [])
                    if contributions:
                        top_contribs = [c for c in contributions[:4] if c.get("contribution", 0) > 0.01]
                        if top_contribs:
                            contrib_data = {
                                c["feature"].replace("_", " ").title(): round(
                                    c["contribution"] * 100, 1
                                )
                                for c in top_contribs
                            }
                            st.bar_chart(contrib_data, height=120)

        # --- Itinerary ---
        with right:
            st.markdown("### Day-by-Day Itinerary")
            itinerary = r.get("itinerary", {})

            if itinerary:
                st.markdown(f"**{itinerary.get('trip_name', '')}**")
                if itinerary.get("overview"):
                    st.caption(itinerary["overview"])

                for day in itinerary.get("days", []):
                    with st.expander(
                        f"Day {day.get('day')} — {day.get('theme', 'Exploration')}",
                        expanded=(day.get("day") == 1),
                    ):
                        if day.get("morning"):
                            st.markdown(f"🌅 **Morning:** {day['morning']}")
                        if day.get("afternoon"):
                            st.markdown(f"☀️ **Afternoon:** {day['afternoon']}")
                        if day.get("evening"):
                            st.markdown(f"🌆 **Evening:** {day['evening']}")
                        if day.get("transport_tip"):
                            st.info(f"🚌 {day['transport_tip']}")
                        if day.get("estimated_daily_budget"):
                            st.caption(f"Est. cost: {day['estimated_daily_budget']}")

                if itinerary.get("practical_tips"):
                    st.markdown("**Practical Tips**")
                    for tip in itinerary["practical_tips"]:
                        st.markdown(f"- {tip}")

                if itinerary.get("total_estimated_cost"):
                    st.info(f"Total estimated cost: {itinerary['total_estimated_cost']}")

        st.divider()

        # --- Final Answer ---
        st.markdown("### TripGenie's Recommendation")
        final_answer = r.get("final_answer", "")
        if final_answer:
            st.markdown(final_answer)

        st.divider()

        # --- Quality & Performance ---
        col_q, col_p = st.columns(2)

        with col_q:
            st.markdown("### Quality Assessment")
            critique = r.get("critique", {})
            if critique:
                qs = critique.get("overall_score", 0)
                label = _score_label(qs)
                cls = _score_color(qs)
                st.markdown(
                    f"**Overall quality:** <span class='{cls}'>{label} ({qs:.0%})</span>",
                    unsafe_allow_html=True,
                )
                checks = {
                    "Budget respected": critique.get("budget_respected", True),
                    "Duration included": critique.get("duration_included", True),
                    "Activities sufficient": critique.get("activities_sufficient", True),
                    "Hotel alignment": critique.get("hotel_alignment", True),
                    "Assumptions stated": critique.get("assumptions_stated", False),
                }
                for check_name, passed in checks.items():
                    icon = "✅" if passed else "⚠️"
                    st.markdown(f"{icon} {check_name}")

                if critique.get("flags"):
                    st.warning("Flags: " + " | ".join(critique["flags"]))
                if critique.get("suggestions"):
                    with st.expander("Suggestions for improvement"):
                        for s in critique["suggestions"]:
                            st.markdown(f"- {s}")

        with col_p:
            st.markdown("### Performance")
            meta = r.get("metadata", {})
            stage_latencies = r.get("stage_latencies", {})

            st.metric("Total latency", f"{r.get('total_latency_ms', 0):.0f} ms")
            st.metric("LLM calls", meta.get("llm_calls", 0))
            st.metric("LLM provider", meta.get("llm_provider", "mock"))
            st.metric("Hotels evaluated", meta.get("candidate_count", 0))

            if stage_latencies:
                st.markdown("**Stage timings (ms)**")
                st.bar_chart(stage_latencies)

# ===========================================================================
# TAB 2: Evaluation
# ===========================================================================
with tab_eval:
    st.markdown("### Offline Evaluation")
    st.markdown(
        "Run the TripGenie pipeline against the built-in sample query set and measure "
        "quality metrics across intent extraction, hotel ranking, itinerary generation, "
        "and final answer helpfulness."
    )

    ecol1, ecol2, ecol3 = st.columns([1, 1, 2])
    with ecol1:
        num_eval_samples = st.slider("Number of queries", 1, 10, 5)
    with ecol2:
        verbose_eval = st.checkbox("Show per-query results", value=True)

    run_eval = st.button("Run Evaluation", type="primary")

    if run_eval:
        eval_result = _call_eval_api(num_eval_samples, verbose_eval)

        if eval_result:
            st.divider()
            st.markdown("### Summary Metrics")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Queries run", eval_result.get("num_queries", 0))
            m2.metric("Avg latency", f"{eval_result.get('avg_latency_ms', 0):.0f} ms")
            m3.metric(
                "Constraint satisfaction",
                f"{eval_result.get('avg_constraint_satisfaction', 0):.0%}",
            )
            m4.metric(
                "Recommendation relevance",
                f"{eval_result.get('avg_recommendation_relevance', 0):.0%}",
            )
            m5.metric(
                "Overall score",
                f"{eval_result.get('avg_overall_score', 0):.0%}",
            )

            st.info(eval_result.get("summary", ""))

            if verbose_eval and eval_result.get("results"):
                st.markdown("### Per-Query Results")
                for qr in eval_result["results"]:
                    status_icon = "✅" if qr.get("error") is None else "❌"
                    with st.expander(
                        f"{status_icon} {qr.get('query_id')} — {qr.get('city', '—')} | "
                        f"Score: {qr.get('overall_score', 0):.0%} | "
                        f"{qr.get('latency_ms', 0):.0f}ms"
                    ):
                        st.markdown(f"*Query:* {qr.get('query', '')}")
                        if qr.get("error"):
                            st.error(f"Error: {qr['error']}")
                        else:
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Constraint sat.", f"{qr.get('constraint_satisfaction', 0):.0%}")
                            mc2.metric("Relevance", f"{qr.get('recommendation_relevance', 0):.0%}")
                            mc3.metric("Completeness", f"{qr.get('itinerary_completeness', 0):.0%}")
                            mc4.metric("Helpfulness", f"{qr.get('helpfulness_score', 0):.0%}")

# ===========================================================================
# TAB 3: About
# ===========================================================================
with tab_about:
    st.markdown("### About TripGenie")
    st.markdown(
        """
TripGenie is an **agentic travel planning assistant** built as a portfolio
project demonstrating GenAI engineering best practices.

**Architecture**

The system uses a multi-stage agent pipeline:

1. **IntentAgent** — Parses natural language queries into structured trip preferences
2. **RetrievalAgent** — Filters candidate hotels and attractions from the dataset
3. **RankingAgent** — Scores hotels using an interpretable ML feature scorer
4. **ItineraryAgent** — Generates day-by-day plans via the LLM layer
5. **CritiqueAgent** — Evaluates plan quality against user requirements
6. **AnswerAgent** — Synthesises the final conversational response

**ML Ranking System**

Hotels are scored using a weighted linear model with features including:
- Budget compatibility
- Location quality score
- Guest review score
- Transport accessibility
- Museum proximity, romantic suitability, family-friendliness, nightlife proximity

Each feature contributes an interpretable score with full contribution breakdown.

**Tech Stack**

FastAPI · Pydantic · LangGraph-inspired custom state machine ·
OpenAI-compatible LLM abstraction · pandas / numpy / scikit-learn ·
Streamlit · Docker · Ruff · mypy · pytest · GitHub Actions

**GitHub**

[github.com/tripgenie](https://github.com/tripgenie-agentic-travel-assistant)
    """
    )
