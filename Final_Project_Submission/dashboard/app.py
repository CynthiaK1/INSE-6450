import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RobustOps", page_icon="🛡", layout="wide")

NAVY = "#1A365D"; TEAL = "#2C7A7B"; SAGE = "#68D391"; ORANGE = "#C05621"; RED = "#C53030"; GREY = "#4A5568"; LIGHT = "#F7FAFC"

st.markdown(f"""
<style>
html, body, .main, .stApp, [data-testid="stAppViewContainer"] {{ background-color: {LIGHT}; }}
h1, h2, h3 {{ color: {NAVY} !important; }}
.stage-card {{ background:white; padding:16px; border-radius:8px; border-left:4px solid #CBD5E0; margin-bottom:8px; min-height:220px; color:#1A202C !important; }}
.stage-card *, .stage-card b, .stage-card i, .stage-card span {{ color:#1A202C !important; }}
.stage-card h4 {{ color:{TEAL} !important; }}
.stage-card i {{ color:{GREY} !important; }}
.stage-card .verdict, .stage-card .verdict * {{ color:inherit !important; }}
.demo-banner, .demo-banner * {{ color:#744210 !important; }}
.stage-active {{ border-left-color:{TEAL}; box-shadow:0 2px 8px rgba(44,122,123,0.15); }}
.stage-pass {{ border-left-color:{SAGE}; }}
.stage-block {{ border-left-color:{ORANGE}; }}
.stage-abstain {{ border-left-color:{RED}; }}
.verdict {{ padding:6px 12px; border-radius:4px; font-weight:700; display:inline-block; }}
.v-pass {{ background:#C6F6D5; color:#22543D; }}
.v-block {{ background:#FEEBC8; color:#7B341E; }}
.v-abstain {{ background:#FED7D7; color:#742A2A; }}
.demo-banner {{ background:#FEFCBF; border-left:4px solid #B7791F; padding:10px 14px; border-radius:4px; font-size:0.85em; color:#744210; }}
</style>
""", unsafe_allow_html=True)

DATA = Path("data")

@st.cache_data
def load_all():
    o = {}
    try: o["diag"] = pd.read_csv(DATA / "diagnosis.csv")
    except Exception: o["diag"] = None
    try: o["mit"] = pd.read_csv(DATA / "mitigation.csv")
    except Exception: o["mit"] = None
    try: o["abl"] = pd.read_csv(DATA / "risk_ablation.csv")
    except Exception: o["abl"] = None
    try:
        with open(DATA / "attribution.json") as f: o["attr"] = json.load(f)
    except Exception: o["attr"] = None
    return o

D = load_all()

if "audit_log" not in st.session_state: st.session_state.audit_log = []
if "current_audit" not in st.session_state: st.session_state.current_audit = None

MODELS = {
    "ResNet-18 / CIFAR-10 (image)": {"key": "image", "acc": 0.856},
    "DistilBERT / SST-2 (NLP)": {"key": "nlp", "acc": 0.905},
    "XGBoost / Adult Income (tabular cls)": {"key": "tab_cls", "acc": 0.878},
    "XGBoost / California Housing (tabular reg)": {"key": "tab_reg", "acc": 0.84},
}
DRIFTS = {"drift1 — mild": "drift1", "drift2 — moderate": "drift2", "drift3 — severe": "drift3"}
MIT_MAP = {
    "covariate": "Replay-CL (80/20 buffer mix) — inputs shifted, relationship intact",
    "label": "Prior correction (no retraining) — recalibrate thresholds only",
    "concept": "ABSTAIN — replay would harm; route to HITL for fresh labels",
    "none": "No action — model is stable on this window",
    "mixed": "Adversarial training on replay buffer — combined regime",
}

def run_audit(mkey, dkey, eps):
    if D["diag"] is None: return None
    row = D["diag"][(D["diag"]["modality"] == mkey) & (D["diag"]["drift"] == dkey)]
    if row.empty: return None
    r = row.iloc[0]
    px, pyx, py, dx = float(r["P(X)"]), float(r["P(Y|X)"]), float(r["P(Y)"]), str(r["diagnosis"])
    asr = min(1.0, max(0.0, (px + pyx) * eps * 2.5))
    da = min(1.0, max(0.0, (px + pyx + py) * 0.4))
    dc = da * 0.7
    risk = 0.35*da + 0.25*dc + 0.25*asr + 0.15*(asr*1.1)
    risk = min(1.0, risk + (0.3 if dx == "concept" else 0.1 if dx != "none" else 0.0))
    tau = 0.15
    if dx == "concept": gate = "ABSTAIN"
    elif risk >= tau: gate = "BLOCK"
    else: gate = "PASS"
    import random
    random.seed(hash(mkey + dkey) % 10000)
    attended = []
    hints = ["age=72 (out of training range)", "income spike +3σ", "categorical=novel value", "feature_5 = NaN", "ratio_2 outside 99th percentile"]
    for i in range(5):
        attended.append({"id": random.randint(1, 5000), "alpha": round(0.05 + random.random()*0.25, 4), "hint": random.choice(hints)})
    return {"ts": datetime.now().strftime("%H:%M:%S"), "model": mkey, "drift": dkey, "eps": eps, "px": px, "pyx": pyx, "py": py, "dx": dx, "da": da, "dc": dc, "asr": asr, "risk": risk, "tau": tau, "gate": gate, "mit": MIT_MAP[dx], "att": attended}

st.title("🛡 RobustOps — AI System Risk Audit")
st.caption("CI/CD pre-deployment gate · Diagnose → Score → Mitigate → Observe")

with st.sidebar:
    st.markdown("### About")
    st.markdown("RobustOps audits ML models for distribution-shift and adversarial risk before deployment, and routes mitigation conditional on the **diagnosed shift type** rather than applying replay-CL blindly.")
    st.markdown("---")
    st.markdown("### Audit history")
    if st.session_state.audit_log:
        for a in reversed(st.session_state.audit_log[-10:]):
            color = {"PASS": SAGE, "BLOCK": ORANGE, "ABSTAIN": RED}[a["gate"]]
            mname = a["model_label"].split(" / ")[0] if "model_label" in a else a["model"]
            st.markdown(f"<div style='font-size:0.8em;border-left:3px solid {color};padding-left:8px;margin:6px 0;'><b>{a['ts']}</b><br>{mname}<br><span style='color:{color};font-weight:700;'>{a['gate']}</span> · risk {a['risk']:.2f}</div>", unsafe_allow_html=True)
    else:
        st.caption("No audits yet")
    st.markdown("---")
    st.caption("INSE 6450 · Cynthia Musila · 40311473")
    st.markdown("[GitHub](https://github.com/CynthiaK1/INSE-6450)")

st.markdown("### 1 · Configure audit")
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    model_label = st.selectbox("Model from registry", list(MODELS.keys()))
with c2:
    drift_label = st.selectbox("Candidate evaluation window", list(DRIFTS.keys()))
with c3:
    eps = st.slider("FGSM ε", 0.0, 0.5, 0.1, 0.05, help="Adversarial threat budget for the stress test")

st.markdown("<div class='demo-banner'>⚠ Demo: drift severity is selectable for illustration. In production, the evaluation window comes from your live traffic monitor (Datadog, Arize, Lakehouse Monitoring, etc.) — there is no dropdown.</div>", unsafe_allow_html=True)
st.markdown("")

if st.button("▶ Run Audit", type="primary"):
    info = MODELS[model_label]
    audit = run_audit(info["key"], DRIFTS[drift_label], eps)
    if audit:
        audit["model_label"] = model_label
        audit["drift_label"] = drift_label
        st.session_state.current_audit = audit
        st.session_state.audit_log.append(audit)
    else:
        st.error("Audit data not found. Make sure data/diagnosis.csv exists.")

a = st.session_state.current_audit
if a is not None:
    st.markdown("---")
    st.markdown("### 2 · Pipeline execution")
    st.caption(f"**{a['model_label']}** · **{a['drift_label']}** · ε = **{a['eps']}** · {a['ts']}")

    cols = st.columns(4)
    dx_class = "v-abstain" if a["dx"] == "concept" else "v-block" if a["dx"] not in ("none",) else "v-pass"
    with cols[0]:
        st.markdown(f"<div class='stage-card stage-active'><h4 style='margin:0;color:{TEAL};'>1 · Diagnose</h4><i style='color:{GREY};font-size:0.8em;'>identify + detect</i><br><br><b>P(X):</b> {a['px']:.4f}<br><b>P(Y|X):</b> {a['pyx']:.4f}<br><b>P(Y):</b> {a['py']:.4f}<br><br>Shift type:<br><span class='verdict {dx_class}'>{a['dx'].upper()}</span></div>", unsafe_allow_html=True)

    gate_class = "stage-pass" if a["gate"] == "PASS" else "stage-block" if a["gate"] == "BLOCK" else "stage-abstain"
    v_class = "v-pass" if a["gate"] == "PASS" else "v-block" if a["gate"] == "BLOCK" else "v-abstain"
    with cols[1]:
        st.markdown(f"<div class='stage-card {gate_class}'><h4 style='margin:0;color:{TEAL};'>2 · Score</h4><i style='color:{GREY};font-size:0.8em;'>decision boundary</i><br><br><b>ΔAcc:</b> {a['da']:.3f}<br><b>ΔConf:</b> {a['dc']:.3f}<br><b>ASR:</b> {a['asr']:.3f}<br><br>Risk: <b>{a['risk']:.3f}</b> (τ={a['tau']})<br><br><span class='verdict {v_class}'>{a['gate']}</span></div>", unsafe_allow_html=True)

    mit_class = "stage-abstain" if "ABSTAIN" in a["mit"] else "stage-pass" if a["gate"] == "PASS" else "stage-block"
    with cols[2]:
        st.markdown(f"<div class='stage-card {mit_class}'><h4 style='margin:0;color:{TEAL};'>3 · Mitigate</h4><i style='color:{GREY};font-size:0.8em;'>protect + respond</i><br><br>Recommended action:<br><br><b>{a['mit']}</b></div>", unsafe_allow_html=True)

    with cols[3]:
        att_html = "".join([f"<div style='font-size:0.75em;border-bottom:1px solid #E2E8F0;padding:4px 0;'><b>#{s['id']}</b> α={s['alpha']}<br><span style='color:{GREY};'>{s['hint']}</span></div>" for s in a["att"]])
        st.markdown(f"<div class='stage-card stage-active'><h4 style='margin:0;color:{TEAL};'>4 · Observe</h4><i style='color:{GREY};font-size:0.8em;'>recover</i><br><br>Top-5 attended samples:{att_html}</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 3 · Operator action")
    b1, b2, b3, _ = st.columns([1, 1, 1.5, 2])
    with b1:
        if st.button("✓ Approve & deploy", disabled=(a["gate"] != "PASS")):
            st.success("Approved. Queued for canary rollout (5% traffic).")
    with b2:
        if st.button("✗ Reject"):
            st.warning("Candidate rejected. Logged to audit trail.")
    with b3:
        if st.button("👤 Send to HITL review"):
            st.info("Routed to human reviewer with top-5 attended samples + feature explanations. Reviewer labels feed back to replay buffer with 2× weight.")

    with st.expander("Why this recommendation? (decision-table lookup)"):
        st.markdown(f"""
The Diagnose pillar identified the shift as **{a['dx']}**. The Mitigate pillar's decision table maps this diagnosis to: **{a['mit']}**

This is the framework's central contribution. The same drift severity calls for **different actions** depending on which kind of shift is happening:

- **Covariate** → retrain via replay-CL (inputs changed, input→output relationship intact)
- **Concept** → abstain (the input→output relationship itself broke; replay would teach stale targets — this is the headline result on California Housing where replay-CL drops R² from 0.69 to 0.44)
- **Label** → recalibrate priors (no retraining needed)
- **None** → deploy as-is

A monitoring tool that only reports "drift detected" cannot make this distinction. RobustOps' diagnosis pillar can.
""")
else:
    st.markdown("---")
    st.info("Configure the audit above and click **Run Audit**. Try **XGBoost / California Housing** + **drift3 — severe** to see the headline ABSTAIN result, or **XGBoost / Adult Income** + **drift1 — mild** to see a clean PASS.")

with st.expander("📊 Raw experimental results (precomputed grid)"):
    t1, t2, t3, t4 = st.tabs(["Diagnosis", "Mitigation 2a/2b", "Risk ablation", "Attribution"])
    with t1:
        if D["diag"] is not None:
            st.dataframe(D["diag"], hide_index=True)
            fp = DATA / "figure3_diagnosis.png"
            if fp.exists(): st.image(str(fp))
    with t2:
        if D["mit"] is not None:
            st.dataframe(D["mit"], hide_index=True)
            st.caption("2a: replay-CL on regression concept-drift fails (R² 0.69 → 0.44). 2b: adv-training on covariate shift wastes compute and degrades accuracy.")
    with t3:
        if D["abl"] is not None:
            st.dataframe(D["abl"].pivot(index="sigma", columns="weights", values="risk").round(3))
            st.caption("All configurations BLOCK at τ=0.15. Variation (0.20–0.83) shows weighting affects severity ranking even when binary decision is unanimous.")
    with t4:
        if D["attr"] is not None:
            ad = D["attr"]
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Attention-MIL recall@50", f"{ad['mil_mean']:.3f}", f"±{ad['mil_std']:.3f}")
            cc2.metric("PSI baseline recall@50", f"{ad['psi_mean']:.3f}", f"±{ad['psi_std']:.3f}")
            cc3.metric("Improvement", f"+{(ad['mil_mean']-ad['psi_mean'])/ad['psi_mean']*100:.0f}%", f"{ad['trials']} trials")

with st.expander("ℹ What's real vs simulated in this demo"):
    st.markdown("""
**Real (from actual experiments):**
- Diagnosis P(X), P(Y|X), P(Y) for each (model, drift) pair come from trained models on real data
- Mitigation 2a/2b verdicts are real measured outcomes
- Attention-MIL vs PSI numbers are real (20-trial average)

**Simulated for the demo:**
- Drift severity is a dropdown; production reads from live monitoring
- Risk score components are computed from diagnosis values via a fast simplified function for UI response time; the real framework runs FGSM and full-window evaluation
- Top-5 attended samples are illustrative placeholders; production returns real sample indices with feature-level attribution

**Production deployment would:**
- Read live windows from monitoring infrastructure (Datadog, Arize, Lakehouse Monitoring)
- Run the full Diagnose pipeline on real evaluation windows
- Persist audit history to a versioned store
- Trigger CI/CD actions (block merge, page on-call, open ticket) on BLOCK or ABSTAIN
""")
