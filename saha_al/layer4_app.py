"""
SAHA-AL Layer 4: Annotation Interface
Dual-mode annotation tool: Batch Mode (spreadsheet) and Focus Mode (1-by-1).
"""

import os
import sys
import streamlit as st
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from saha_al.config import (
    ANNOTATION_QUEUE_PATH,
    GOLD_STANDARD_PATH,
    SKIPPED_PATH,
    FLAGGED_PATH,
)
from saha_al.utils.io_helpers import read_jsonl, write_jsonl, append_jsonl
from saha_al.utils.quality_checks import check_leakage

st.set_page_config(
    page_title="SAHA-AL Annotation Pipeline",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Data editor table full-height rows */
[data-testid="stDataFrame"] div[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
/* Metric values */
div[data-testid="stMetricValue"] {
    font-size: 2rem;
    color: #8B5CF6;
    font-weight: 800;
}
/* Info box tweak */
.stAlert p { font-size: 0.9rem; }
/* Radio inline */ 
div[role="radiogroup"] { flex-direction: row; gap: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def init_session_state():
    if "queue" not in st.session_state:
        st.session_state.queue = read_jsonl(ANNOTATION_QUEUE_PATH)
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "stats" not in st.session_state:
        st.session_state.stats = {"accepted": 0, "flagged": 0, "skipped": 0}
    # Focus mode: entry offset within current batch view
    if "focus_entry" not in st.session_state:
        st.session_state.focus_entry = None


def save_progress():
    remaining = st.session_state.queue[st.session_state.current_idx:]
    write_jsonl(ANNOTATION_QUEUE_PATH, remaining)


def load_gold_count():
    if os.path.exists(GOLD_STANDARD_PATH):
        return len(read_jsonl(GOLD_STANDARD_PATH))
    return 0


def get_default_rewrite(text, entities):
    if not entities:
        return text
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0), reverse=True)
    new_text = text
    for ent in sorted_ents:
        etype = ent.get("label", "UNKNOWN")
        s, e = ent.get("start", 0), ent.get("end", 0)
        new_text = new_text[:s] + f"[REDACTED_{etype}]" + new_text[e:]
    return new_text


def get_highlighted_html(text, entities):
    if not entities:
        return text
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0), reverse=True)
    html = text
    for ent in sorted_ents:
        s, e = ent.get("start", 0), ent.get("end", 0)
        label = ent.get("label", "?")
        value = ent.get("value", html[s:e])
        tag = (
            f'<span style="background:#fbbf24;color:#000;padding:1px 5px;'
            f'border-radius:4px;font-weight:700;">{value}</span>'
            f'<sup style="color:#ef4444;font-size:0.75em;font-weight:700;">{label}</sup>'
        )
        html = html[:s] + tag + html[e:]
    return html


def fmt_entities(entities):
    if not entities:
        return "—"
    return " · ".join([f"{e['value']} [{e['label']}]" for e in entities])


def persist_results(accept_list, flag_list, skip_list):
    if accept_list:
        append_jsonl(GOLD_STANDARD_PATH, accept_list)
        st.session_state.stats["accepted"] += len(accept_list)
    if flag_list:
        append_jsonl(FLAGGED_PATH, flag_list)
        st.session_state.stats["flagged"] += len(flag_list)
    if skip_list:
        append_jsonl(SKIPPED_PATH, skip_list)
        st.session_state.stats["skipped"] += len(skip_list)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar(mode):
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:12px 0 6px;">
                <span style="font-size:2.6rem;font-weight:900;
                    background:linear-gradient(90deg,#8B5CF6,#EC4899);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    SAHA‑AL
                </span><br>
                <span style="font-size:0.75rem;letter-spacing:3px;color:#6B7280;
                    text-transform:uppercase;">Annotation Engine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        total = len(st.session_state.queue)
        done = st.session_state.current_idx
        pct = done / max(total, 1)
        st.progress(pct, text=f"{done:,} / {total:,} done  ({int(pct*100)}%)")

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Remaining", f"{max(0, total-done):,}")
        c2.metric("Gold ✅", f"{load_gold_count():,}")

        st.divider()
        st.markdown("**Session**")
        st.write(f"✅ Accepted  — **{st.session_state.stats['accepted']}**")
        st.write(f"🚩 Flagged   — **{st.session_state.stats['flagged']}**")
        st.write(f"⏭ Skipped   — **{st.session_state.stats['skipped']}**")

        st.divider()
        # ── Settings ──────────────────────────────────
        st.markdown("**⚙️ Settings**")
        if mode == "batch":
            bs = st.slider("Batch size", 5, 50, 20, 5, key="batch_size")
        else:
            bs = st.session_state.get("batch_size", 20)

        st.divider()
        st.markdown("**👤 Annotator**")
        annotator = st.selectbox(
            "Select your annotator ID",
            options=["A1 — Annotator 1", "A2 — Annotator 2", "A3 — Annotator 3"],
            key="annotator_id",
            label_visibility="collapsed",
        )
        annotator_code = annotator.split(" ")[0]  # e.g. "A1"

        if st.button("↩ Reset queue pointer", use_container_width=True):
            st.session_state.current_idx = 0
            st.session_state.focus_entry = None
            st.rerun()

        return bs, annotator_code


# ─── Batch Mode ──────────────────────────────────────────────────────────────

def render_batch_mode(batch_size, annotator_id):
    total = len(st.session_state.queue)
    cur = st.session_state.current_idx
    end = min(cur + batch_size, total)
    batch = st.session_state.queue[cur:end]

    # Gradient header
    st.markdown(
        f"""<h2 style="font-weight:800;background:linear-gradient(90deg,#8B5CF6,#EC4899);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        📋 Batch #{cur//batch_size + 1} &nbsp;·&nbsp; Entries {cur}–{end-1}</h2>""",
        unsafe_allow_html=True,
    )

    # ── Pagination controls ────────────────────────────────────────────────────
    nav_l, nav_info, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("◀ Prev Batch", disabled=(cur == 0), use_container_width=True):
            st.session_state.current_idx = max(0, cur - batch_size)
            st.rerun()
    with nav_info:
        total_batches = max(1, (total + batch_size - 1) // batch_size)
        current_batch = cur // batch_size + 1
        st.markdown(
            f"<p style='text-align:center;color:#9CA3AF;margin:0;padding-top:6px;'>"
            f"Batch {current_batch} of {total_batches}</p>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button("Next Batch ▶", disabled=(end >= total), use_container_width=True):
            st.session_state.current_idx = end
            st.rerun()

    st.divider()
    st.caption("💡 Edit **Anonymized Rewrite** cells. Set **Action** to Flag or Skip if needed. Hit **Save Batch** when done.")

    # Keyboard shortcut
    components.html(
        """<script>
        const doc = window.parent.document;
        doc.addEventListener('keydown', e => {
            if ((e.ctrlKey||e.metaKey) && e.key==='Enter') {
                const btn = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.includes('Save Batch'));
                if (btn) btn.click();
            }
        });
        </script>""",
        height=0, width=0,
    )

    # Build DataFrame
    rows = []
    for i, entry in enumerate(batch):
        rows.append({
            "_i": i,
            "ID": entry.get("entry_id", "?"),
            "Original Text": entry.get("original_text", ""),
            "PII Found": fmt_entities(entry.get("entities", [])),
            "Anonymized Rewrite ✏️": get_default_rewrite(
                entry.get("original_text", ""), entry.get("entities", [])
            ),
            "Action": "Accept",
        })
    df = pd.DataFrame(rows)

    edited_df = st.data_editor(
        df,
        column_config={
            "_i": None,
            "ID": st.column_config.NumberColumn(disabled=True, width=60),
            "Original Text": st.column_config.TextColumn(disabled=True, width="large", max_chars=None),
            "PII Found": st.column_config.TextColumn(disabled=True, width="medium"),
            "Anonymized Rewrite ✏️": st.column_config.TextColumn(required=True, width="large"),
            "Action": st.column_config.SelectboxColumn(
                options=["Accept", "Flag", "Skip"],
                required=True,
                width=100,
            ),
        },
        hide_index=True,
        use_container_width=True,
        height=min(38 + len(batch) * 42, 900),   # dynamic row height
        key=f"editor_{cur}_{end}",
    )

    # ── Save button ────────────────────────────────────────────────────────────
    save_col, focus_col = st.columns([3, 1])
    with save_col:
        save_clicked = st.button("✅ Save Batch  (Ctrl+Enter)", type="primary", use_container_width=True)
    with focus_col:
        jump_target = st.number_input("Jump to entry #", value=cur, min_value=0,
                                       max_value=total - 1, step=1, key="jump_input")
        if st.button("↩ Jump", use_container_width=True):
            st.session_state.current_idx = jump_target
            st.rerun()

    if save_clicked:
        accept_list, flag_list, skip_list = [], [], []
        errors = []
        for _, row in edited_df.iterrows():
            i = row["_i"]
            orig = batch[i]
            action = row["Action"]
            rewrite = str(row["Anonymized Rewrite ✏️"]).strip()
            proc = orig.copy()
            proc["timestamp"] = datetime.now().isoformat()
            proc["annotator_id"] = annotator_id
            if action == "Accept":
                if not rewrite:
                    errors.append(f"Entry {orig['entry_id']}: rewrite is empty.")
                    continue
                leaks = check_leakage(orig["original_text"], rewrite, orig.get("entities", []))
                if leaks:
                    errors.append(f"Entry {orig['entry_id']}: leakage — {', '.join(leaks)}")
                    continue
                proc["anonymized_text"] = rewrite
                accept_list.append(proc)
            elif action == "Flag":
                proc["reason"] = "Flagged in batch"
                flag_list.append(proc)
            else:
                skip_list.append(proc)

        if errors:
            for e in errors:
                st.error(e)
        else:
            persist_results(accept_list, flag_list, skip_list)
            st.session_state.current_idx = end
            save_progress()
            st.toast(f"Saved {len(accept_list)} ✅  {len(flag_list)} 🚩  {len(skip_list)} ⏭", icon="🔥")
            st.rerun()


# ─── Focus Mode ────────────────────────────────────────────────────────────────

def render_focus_mode(annotator_id):
    total = len(st.session_state.queue)
    cur = st.session_state.current_idx
    entry = st.session_state.queue[cur]

    # ── Navigation ────────────────────────────────────────────────────────────
    nav_l, nav_c, nav_r = st.columns([1, 6, 1])
    with nav_l:
        if st.button("◀ Prev", disabled=(cur == 0), use_container_width=True):
            st.session_state.current_idx -= 1
            st.rerun()
    with nav_c:
        # Jump to arbitrary entry
        go_to = st.number_input(
            "Jump to entry index",
            value=cur, min_value=0, max_value=total - 1,
            step=1, label_visibility="collapsed", key="focus_jump",
        )
        if go_to != cur:
            st.session_state.current_idx = go_to
            st.rerun()
        st.markdown(
            f"<p style='text-align:center;color:#9CA3AF;font-size:0.85rem;'>"
            f"Entry {cur + 1} of {total} &nbsp;·&nbsp; ID #{entry.get('entry_id','?')}</p>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button("Next ▶", disabled=(cur >= total - 1), use_container_width=True):
            st.session_state.current_idx += 1
            st.rerun()

    st.divider()

    # ── Content layout ──────────────────────────────────────────────────────
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("📄 Original Text")
        highlighted = get_highlighted_html(entry.get("original_text", ""), entry.get("entities", []))
        st.markdown(
            f'<div style="background:#111827;color:#F9FAFB;padding:16px;border-radius:10px;'
            f'line-height:1.8;font-size:1.05rem;border:1px solid #374151;">{highlighted}</div>',
            unsafe_allow_html=True,
        )

        if entry.get("masked_text"):
            st.subheader("🔖 Masked Reference")
            st.code(entry["masked_text"], language=None)

        if entry.get("entities"):
            st.subheader("🏷️ Detected Entities")
            ent_df = pd.DataFrame([
                {"Label": e["label"], "Value": e["value"]}
                for e in entry["entities"]
            ])
            st.dataframe(ent_df, hide_index=True, use_container_width=True)

    with right:
        st.subheader("✍️ Your Rewrite")
        default = get_default_rewrite(entry.get("original_text", ""), entry.get("entities", []))
        rewrite = st.text_area(
            "Pseudonymize — replace every [REDACTED_*] with a realistic fake:",
            value=default,
            height=220,
            key=f"focus_rewrite_{cur}",
        ).strip()

        # Keyboard shortcut hint
        components.html(
            """<script>
            const doc = window.parent.document;
            doc.addEventListener('keydown', e => {
                if ((e.ctrlKey||e.metaKey) && e.key==='Enter') {
                    const btn = Array.from(doc.querySelectorAll('button'))
                        .find(b => b.innerText.includes('Accept'));
                    if (btn) btn.click();
                }
            });
            </script>""",
            height=0, width=0,
        )

        st.caption("⌨️  Ctrl+Enter = Accept")

        btn1, btn2, btn3 = st.columns(3)
        proc = entry.copy()
        proc["timestamp"] = datetime.now().isoformat()
        proc["annotator_id"] = annotator_id

        with btn1:
            if st.button("✅ Accept", type="primary", use_container_width=True, key=f"acc_{cur}"):
                if not rewrite:
                    st.error("Rewrite is empty.")
                else:
                    leaks = check_leakage(entry["original_text"], rewrite, entry.get("entities", []))
                    if leaks:
                        st.error(f"PII leakage: {', '.join(leaks)}")
                    else:
                        proc["anonymized_text"] = rewrite
                        persist_results([proc], [], [])
                        st.session_state.current_idx += 1
                        save_progress()
                        st.toast("Saved! ✅", icon="🔥")
                        st.rerun()

        with btn2:
            if st.button("🚩 Flag", use_container_width=True, key=f"flag_{cur}"):
                proc["reason"] = "Annotator flagged"
                persist_results([], [proc], [])
                st.session_state.current_idx += 1
                save_progress()
                st.rerun()

        with btn3:
            if st.button("⏭ Skip", use_container_width=True, key=f"skip_{cur}"):
                persist_results([], [], [proc])
                st.session_state.current_idx += 1
                save_progress()
                st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_session_state()

    # Mode toggle at top of page
    mode = st.radio(
        "**Annotation Mode**",
        ["📋 Batch Mode", "🔍 Focus Mode"],
        horizontal=True,
        key="annotation_mode",
    )

    mode_key = "batch" if "Batch" in mode else "focus"
    batch_size, annotator_id = render_sidebar(mode_key)

    st.divider()

    if st.session_state.current_idx >= len(st.session_state.queue):
        st.balloons()
        st.success("🎉 Annotation queue is complete!")
        return

    if mode_key == "batch":
        render_batch_mode(batch_size, annotator_id)
    else:
        render_focus_mode(annotator_id)


if __name__ == "__main__":
    main()
