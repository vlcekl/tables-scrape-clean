import json
from typing import Dict, Any, Tuple, Optional

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from typing import Literal


# ---------------------------------------------------------------------
# Shared Pydantic model (single source of truth)
# ---------------------------------------------------------------------


class FormData(BaseModel):
    project_name: str = Field(..., min_length=1)
    requester_name: str = Field(..., min_length=1)
    data_type: Literal["Select...", "Genomic data", "Clinical data", "Imaging data", "Other"] = "Select..."
    sensitivity: Literal[
        "Select...",
        "Non-sensitive aggregate data",
        "De-identified data",
        "Sensitive personal data",
    ] = "Select..."
    duration: Literal["3 months", "6 months", "12 months", "24 months"] = "3 months"
    irb_reference: Optional[str] = None
    justification: str = Field(..., min_length=1)


def validate_form_dict(data: dict) -> Tuple[Optional[FormData], Dict[str, str]]:
    """Validate a plain dict against FormData.
    Returns (model_or_None, field_errors).
    """
    try:
        model = FormData(**data)
    except ValidationError as e:
        errors: Dict[str, str] = {}
        for err in e.errors():
            loc = err["loc"][0]
            msg = err["msg"]
            errors[loc] = msg
        return None, errors

    # Extra conditional rule: IRB required when sensitivity is "Sensitive personal data"
    errors: Dict[str, str] = {}
    if model.sensitivity == "Sensitive personal data":
        if not (model.irb_reference and model.irb_reference.strip()):
            errors["irb_reference"] = "IRB / ethics reference is required for sensitive personal data."

    if errors:
        return None, errors

    return model, {}


# ---------------------------------------------------------------------
# Mock agent – replace `run_ar_agent` with your real LangGraph AR_agent
# ---------------------------------------------------------------------


def run_ar_agent(message: str) -> str:
    """Mock AR_agent function.

    In your real setup, you'd call your LangGraph graph here, passing this
    `message: str` and getting back `reply: str`.

    This mock just:
    - Recognizes FORM_UPDATE messages and quietly acknowledges them.
    - For USER_CHAT, it reads the embedded form JSON and gives a simple reply.
    """
    if message.startswith("[FORM_UPDATE]"):
        # In reality, the agent would update its internal memory with the form state here.
        return "Form update processed."

    if message.startswith("[USER_CHAT]"):
        # Extract form_state_json block
        try:
            start = message.index("```form_state_json") + len("```form_state_json")
            end = message.index("```", start)
            form_block = message[start:end].strip()
            form_state = json.loads(form_block)
        except Exception:
            form_state = {}

        # Simple behavior: show snapshot
        lines = [
            "Thanks for your message. Here's what I see in the current form:",
            f"- Project name: {form_state.get('project_name') or '—'}",
            f"- Requester: {form_state.get('requester_name') or '—'}",
            f"- Data type: {form_state.get('data_type') or '—'}",
            f"- Sensitivity: {form_state.get('sensitivity') or '—'}",
            f"- Duration: {form_state.get('duration') or '—'}",
            "",
            "Ask me to review your form or explain any field.",
        ]
        return "\n".join(lines)

    # Fallback
    return "I received a message but could not determine its type."


# ---------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------


def init_state() -> None:
    defaults = {
        "project_name": "",
        "requester_name": "",
        "data_type": "Select...",
        "sensitivity": "Select...",
        "justification": "",
        "duration": "3 months",
        "irb_reference": "",
        "chat_history": [],
        "validation_errors": {},
        "last_form_snapshot": None,  # to detect changes
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_form_state_from_session() -> Dict[str, Any]:
    return {
        "project_name": st.session_state.project_name,
        "requester_name": st.session_state.requester_name,
        "data_type": st.session_state.data_type,
        "sensitivity": st.session_state.sensitivity,
        "duration": st.session_state.duration,
        "irb_reference": st.session_state.irb_reference,
        "justification": st.session_state.justification,
    }


def validate_current_form() -> Tuple[Optional[FormData], Dict[str, str]]:
    form_dict = build_form_state_from_session()
    model, errors = validate_form_dict(form_dict)
    return model, errors


def build_form_update_message(form_model: Optional[FormData], errors: Dict[str, str]) -> str:
    """Special message format used when the form changes.

    The agent can recognize `[FORM_UPDATE]` and use the JSON blocks
    to update its internal state.
    """
    if form_model is not None:
        form_json = form_model.model_dump()
    else:
        form_json = build_form_state_from_session()

    return f"""[FORM_UPDATE]
The user has just updated the form. Here is the latest validated state and errors.

```form_state_json
{json.dumps(form_json, indent=2)}
```

```validation_errors_json
{json.dumps(errors, indent=2)}
```""".strip()


def build_user_chat_message(form_model: Optional[FormData], errors: Dict[str, str], user_text: str) -> str:
    """Special message format used when the user chats.

    The agent can recognize `[USER_CHAT]` and see both:
    - current form state
    - current validation errors
    - the user's natural language message
    """
    if form_model is not None:
        form_json = form_model.model_dump()
    else:
        form_json = build_form_state_from_session()

    return f"""[USER_CHAT]
You are assisting with a data access request form.

Here is the current form state:

```form_state_json
{json.dumps(form_json, indent=2)}
```

Here are the current validation errors based on our shared Pydantic model:

```validation_errors_json
{json.dumps(errors, indent=2)}
```

User message:
\"\"\"{user_text}\"\"\"
""".strip()


def compute_progress(errors: Dict[str, str]) -> Tuple[float, str]:
    required_fields = [
        "project_name",
        "requester_name",
        "data_type",
        "sensitivity",
        "justification",
    ]
    if st.session_state.sensitivity == "Sensitive personal data":
        required_fields.append("irb_reference")

    filled = 0
    for field in required_fields:
        value = st.session_state.get(field, "")
        if isinstance(value, str):
            if value.strip() and value not in ("Select...",):
                filled += 1
        else:
            if value:
                filled += 1

    total = len(required_fields)
    ratio = filled / total if total else 0.0
    label = f"Form completion: {filled}/{total} required fields"
    return ratio, label


def field_error(field_key: str, errors: Dict[str, str]) -> None:
    if field_key in errors:
        st.markdown(
            f"<span style='color:#E53E3E; font-size: 0.85rem;'>{errors[field_key]}</span>",
            unsafe_allow_html=True,
        )


def render_progress_header(errors: Dict[str, str]) -> None:
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


def render_form(errors: Dict[str, str]) -> None:
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


def render_chat(form_model: Optional[FormData], errors: Dict[str, str]) -> None:
    st.markdown("### Assistant")
    st.caption("Your chat messages are sent together with the current form state and validation errors.")

    completion_pct = int(compute_progress(errors)[0] * 100)
    with st.container():
        st.markdown(
            "<div style='padding:0.75rem 1rem; border-radius:0.75rem; "
            "background-color:#EDF2F7; font-size:0.85rem;'>"
            f"<b>Current form status</b><br/>Completion: {completion_pct}%<br/>"
            f"Open issues: {len(errors)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_message = st.chat_input("Type your question or instruction...")
    if user_message:
        # Store user message
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Build USER_CHAT message and send to agent
        agent_message = build_user_chat_message(form_model, errors, user_message)
        reply = run_ar_agent(agent_message)

        # Display and store reply
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)


def render_footer(errors: Dict[str, str]) -> None:
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
        st.caption("You can also ask the assistant to review your form.")

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
    st.set_page_config(page_title="Data Access Request – Pydantic + Agent", layout="wide")
    init_state()

    # Capture old snapshot to detect changes
    old_snapshot = st.session_state.last_form_snapshot

    # We first compute errors based on current state (before rendering chat).
    form_model, errors = validate_current_form()
    st.session_state.validation_errors = errors

    # Layout: form left, chat right
    col_form, col_chat = st.columns([2, 1], gap="large")

    with col_form:
        render_form(errors)

    # Rebuild form state after possible widget changes
    form_model, errors = validate_current_form()
    st.session_state.validation_errors = errors
    new_snapshot = build_form_state_from_session()
    st.session_state.last_form_snapshot = new_snapshot

    # If the form changed since last run, send a FORM_UPDATE message to the agent
    if old_snapshot is not None and old_snapshot != new_snapshot:
        update_message = build_form_update_message(form_model, errors)
        # In real deployment, the agent would update its memory here.
        _ = run_ar_agent(update_message)  # reply ignored; this is just a sync signal

    with col_chat:
        render_chat(form_model, errors)

    render_footer(errors)


if __name__ == "__main__":
    main()