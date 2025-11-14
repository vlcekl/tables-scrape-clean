import re
from typing import Dict, List, Tuple

import streamlit as st


# ---------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------
def init_state() -> None:
    """Initialize Streamlit session state keys used in the app."""
    default_values = {
        "project_name": "",
        "requester_name": "",
        "data_type": "Select...",
        "sensitivity": "Select...",
        "justification": "",
        "duration": "3 months",
        "irb_reference": "",
        "chat_history": [],  # list[{"role": "user"|"assistant", "content": str}]
        "validation_errors": {},
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------
# Form validation & progress
# ---------------------------------------------------------------------
def validate_form() -> Dict[str, str]:
    """
    Validate current form values in session_state.
    Returns mapping field_key -> error_message.
    """
    errors: Dict[str, str] = {}

    if not st.session_state.project_name.strip():
        errors["project_name"] = "Please provide the project name."

    if not st.session_state.requester_name.strip():
        errors["requester_name"] = "Please provide your name."

    if st.session_state.data_type == "Select...":
        errors["data_type"] = "Please select the type of data you need."

    if st.session_state.sensitivity == "Select...":
        errors["sensitivity"] = "Please select the data sensitivity level."

    if not st.session_state.justification.strip():
        errors["justification"] = "Please explain why you need this data."

    if st.session_state.sensitivity == "Sensitive personal data" and not st.session_state.irb_reference.strip():
        errors["irb_reference"] = "IRB / ethics approval reference is required for sensitive personal data."

    return errors


def compute_progress(errors: Dict[str, str]) -> Tuple[float, str]:
    """
    Compute completion ratio based on required fields.
    Returns (ratio, label_text).
    """
    required_fields = [
        "project_name",
        "requester_name",
        "data_type",
        "sensitivity",
        "justification",
    ]
    # IRB is conditionally required
    if st.session_state.sensitivity == "Sensitive personal data":
        required_fields.append("irb_reference")

    filled = 0
    for field in required_fields:
        value = st.session_state.get(field, "")
        if isinstance(value, str):
            if value.strip():
                filled += 1
        else:
            if value:
                filled += 1

    total = len(required_fields)
    ratio = filled / total if total > 0 else 0.0
    label = f"Form completion: {filled}/{total} required fields"
    return ratio, label


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
def field_error(field_key: str, errors: Dict[str, str]) -> None:
    """Render a small red error text under a field if needed."""
    if field_key in errors:
        st.markdown(
            f"<span style='color:#E53E3E; font-size: 0.85rem;'>{errors[field_key]}</span>",
            unsafe_allow_html=True,
        )


def render_progress_header(errors: Dict[str, str]) -> None:
    """Render a simple progress indicator for the steps."""
    st.markdown("### Data Access Request")
    steps = ["Project info", "Data request", "Justification", "Review & submit"]
    cols = st.columns(len(steps))

    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"<div style='font-size:0.8rem; text-transform:uppercase; "
                f"letter-spacing:0.05em; color:#4A5568;'>{step}</div>",
                unsafe_allow_html=True,
            )

    progress, label = compute_progress(errors)
    st.progress(progress)
    st.caption(label)


# ---------------------------------------------------------------------
# Form rendering (left pane)
# ---------------------------------------------------------------------
def render_form(errors: Dict[str, str]) -> None:
    """Render the main form area, split in logical sections."""
    render_progress_header(errors)

    # Section 1: Project info
    with st.container():
        st.markdown("#### 1. Project info")
        st.text_input(
            "Project name",
            key="project_name",
            placeholder="e.g. Phoenix – Genomic modeling",
        )
        field_error("project_name", errors)

        st.text_input(
            "Your name",
            key="requester_name",
            placeholder="e.g. Dr. Jane Doe",
        )
        field_error("requester_name", errors)

    st.divider()

    # Section 2: Data request
    with st.container():
        st.markdown("#### 2. Data requested")

        st.selectbox(
            "Type of data",
            key="data_type",
            options=["Select...", "Genomic data", "Clinical data", "Imaging data", "Other"],
        )
        field_error("data_type", errors)

        st.selectbox(
            "Data sensitivity level",
            key="sensitivity",
            options=[
                "Select...",
                "Non-sensitive aggregate data",
                "De-identified data",
                "Sensitive personal data",
            ],
        )
        field_error("sensitivity", errors)

        st.selectbox(
            "Requested access duration",
            key="duration",
            options=["3 months", "6 months", "12 months", "24 months"],
        )

        if st.session_state.sensitivity == "Sensitive personal data":
            st.text_input(
                "IRB / Ethics approval reference",
                key="irb_reference",
                placeholder="e.g. IRB-2025-1234",
            )
            field_error("irb_reference", errors)

    st.divider()

    # Section 3: Justification
    with st.container():
        st.markdown("#### 3. Justification")
        st.text_area(
            "Why do you need this data?",
            key="justification",
            height=140,
            placeholder=(
                "Describe the purpose of the data access, expected outcomes, and how you will "
                "protect the data."
            ),
        )
        field_error("justification", errors)


# ---------------------------------------------------------------------
# Assistant logic (right pane)
# ---------------------------------------------------------------------
def summarize_form_for_assistant() -> str:
    """Return a compact summary of the current form state."""
    lines: List[str] = [
        f"Project name: {st.session_state.project_name or '—'}",
        f"Requester: {st.session_state.requester_name or '—'}",
        f"Data type: {st.session_state.data_type}",
        f"Sensitivity: {st.session_state.sensitivity}",
        f"Duration: {st.session_state.duration}",
    ]
    if st.session_state.sensitivity == "Sensitive personal data":
        irb = st.session_state.irb_reference or "—"
        lines.append(f"IRB reference: {irb}")
    if st.session_state.justification.strip():
        lines.append("Justification: present")
    else:
        lines.append("Justification: missing")
    return "\n".join(lines)


def parse_set_command(message: str) -> List[str]:
    """
    Very simple parser for commands like:
    - 'set project name to Phoenix project'
    - 'set data type to genomic data'
    Returns list of updates applied (for user feedback).
    """
    text = message.lower()
    updates_applied: List[str] = []

    patterns = {
        "project_name": r"set\s+project\s+name\s+to\s+(.+)",
        "requester_name": r"set\s+(?:my\s+name|requester\s+name)\s+to\s+(.+)",
        "data_type": r"set\s+data\s+type\s+to\s+(.+)",
        "sensitivity": r"set\s+sensitivity\s+to\s+(.+)",
        "duration": r"set\s+duration\s+to\s+(.+)",
        "irb_reference": r"set\s+irb\s+(?:ref|reference)?\s*to\s+(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            # Normalize some common choices to match selectbox options
            if key == "data_type":
                normalized = value.lower()
                if "genomic" in normalized:
                    value = "Genomic data"
                elif "clinical" in normalized:
                    value = "Clinical data"
                elif "image" in normalized:
                    value = "Imaging data"
                else:
                    value = "Other"
            if key == "sensitivity":
                normalized = value.lower()
                if "non" in normalized and "aggregate" in normalized:
                    value = "Non-sensitive aggregate data"
                elif "de-identified" in normalized or "deidentified" in normalized:
                    value = "De-identified data"
                elif "sensitive" in normalized:
                    value = "Sensitive personal data"
            if key == "duration":
                normalized = value.lower()
                if "3" in normalized:
                    value = "3 months"
                elif "6" in normalized:
                    value = "6 months"
                elif "12" in normalized or "1 year" in normalized:
                    value = "12 months"
                elif "24" in normalized or "2 years" in normalized:
                    value = "24 months"

            st.session_state[key] = value
            updates_applied.append(key)

    return updates_applied


def assistant_reply(user_message: str, errors: Dict[str, str]) -> str:
    """
    Very simple rule-based 'agent' that is context-aware:
    - Can set form fields from commands
    - Can review missing/invalid fields
    - Can explain fields
    - Uses current form state in answers
    """
    msg_lower = user_message.lower().strip()

    # 1. Try to apply "set ..." commands
    updates = parse_set_command(user_message)
    if updates:
        field_names = {
            "project_name": "project name",
            "requester_name": "your name",
            "data_type": "data type",
            "sensitivity": "sensitivity level",
            "duration": "access duration",
            "irb_reference": "IRB reference",
        }
        updated_labels = [field_names[u] for u in updates]
        return (
            "Got it. I updated: "
            + ", ".join(updated_labels)
            + ".\n\nYou can say “review my request” to check if anything is still missing."
        )

    # 2. Review request
    if "review" in msg_lower or "check" in msg_lower or "ready to submit" in msg_lower:
        if not errors:
            return (
                "I've checked your request and everything looks complete.\n\n"
                "You can submit it now. Once submitted, I can initiate the data access approval workflow."
            )
        else:
            missing = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in errors.items())
            return (
                "Here’s what still needs attention before you submit:\n\n"
                f"{missing}\n\n"
                "Once you fill these in, ask me to review again."
            )

    # 3. Explain fields / guidance
    if "why do i need" in msg_lower or "what is" in msg_lower:
        if "justification" in msg_lower:
            return (
                "The justification explains why you need this data and how it will be used. "
                "Approvers use it to evaluate whether the request is necessary and proportionate. "
                "Try to be specific, mention your project goals, and how you will safeguard the data."
            )
        if "irb" in msg_lower or "ethics" in msg_lower:
            return (
                "An IRB / Ethics approval reference is required when you request sensitive personal data. "
                "It indicates that an independent committee has reviewed the risks and approved your protocol."
            )
        if "sensitivity" in msg_lower:
            return (
                "The sensitivity level describes how risky the data would be if misused. "
                "Non-sensitive aggregate data is highly aggregated and anonymized. "
                "De-identified data has personal identifiers removed. "
                "Sensitive personal data includes health, genetic, or other data that can strongly impact privacy."
            )

    # 4. Summarize current form if user asks "what have i entered" or similar
    if "what have i entered" in msg_lower or "show summary" in msg_lower or "summary" in msg_lower:
        summary = summarize_form_for_assistant()
        return "Here’s a summary of your current request:\n\n" + summary

    # 5. Default: polite, context-aware fallback
    summary = summarize_form_for_assistant()
    base_answer = (
        "I'm aware of the information you've entered so far. "
        "You can ask me to:\n"
        "- Review your request for missing or inconsistent fields\n"
        "- Help you choose a sensitivity level or duration\n"
        "- Draft a strong justification\n"
        "- Update any field (e.g. “set project name to …”)\n\n"
        "Here’s your current request snapshot:\n\n"
        f"{summary}"
    )
    return base_answer


def render_chat(errors: Dict[str, str]) -> None:
    """Render the assistant pane with chat history and input."""
    st.markdown("### Assistant")
    st.caption("Ask questions, get explanations, or tell me to update the form for you.")

    # Small context card
    with st.container():
        st.markdown(
            "<div style='padding:0.75rem 1rem; border-radius:0.75rem; "
            "background-color:#EDF2F7; font-size:0.85rem;'>"
            "<b>Current form status</b><br/>"
            f"Completion: {int(compute_progress(errors)[0] * 100)}%<br/>"
            f"Open issues: {len(errors)}"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacing

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_message = st.chat_input("Type your question or instruction...")
    if user_message:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Generate assistant reply
        reply = assistant_reply(user_message, errors)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Display immediately
        with st.chat_message("assistant"):
            st.write(reply)


# ---------------------------------------------------------------------
# Footer / submission bar
# ---------------------------------------------------------------------
def render_footer(errors: Dict[str, str]) -> None:
    """Render a fixed-style footer summarizing status and submit action."""
    st.divider()
    col_status, col_actions = st.columns([3, 2])

    with col_status:
        if errors:
            st.markdown(
                "<span style='color:#E53E3E; font-weight:500;'>"
                f"Not ready to submit – {len(errors)} issues need attention.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#38A169; font-weight:500;'>"
                "Ready to submit – all required fields are complete.</span>",
                unsafe_allow_html=True,
            )
        st.caption("You can also ask the assistant to review your request.")

    with col_actions:
        submit_disabled = bool(errors)
        if st.button("Submit request", disabled=submit_disabled, use_container_width=True):
            st.success(
                "Your request has been submitted. The approval workflow has been initiated. "
                "You will be notified once a decision is made."
            )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Data Access Request – Form & Assistant",
        layout="wide",
    )

    init_state()

    # Use last-known errors for inline display
    previous_errors = st.session_state.get("validation_errors", {})

    col_form, col_chat = st.columns([2, 1], gap="large")

    with col_form:
        render_form(previous_errors)

    # Re-validate after the user might have changed inputs
    current_errors = validate_form()
    st.session_state["validation_errors"] = current_errors

    with col_chat:
        render_chat(current_errors)

    render_footer(current_errors)


if __name__ == "__main__":
    main()
