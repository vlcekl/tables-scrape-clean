import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import streamlit as st


# ---------------------------------------------------------------------
# Mock schema module (replace with your real validation module)
# ---------------------------------------------------------------------


@dataclass
class FormSchema:
    """
    Mock of your schema/validation module.

    In your real project, you'd import your schema here and expose
    a similar `validate` interface.
    """

    required_base_fields: List[str] = field(
        default_factory=lambda: [
            "project_name",
            "requester_name",
            "data_type",
            "sensitivity",
            "justification",
        ]
    )

    def validate(self, form_state: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate form_state and return field -> error message.
        """
        errors: Dict[str, str] = {}

        # Basic required checks
        for field in self.required_base_fields:
            if not str(form_state.get(field, "")).strip():
                errors[field] = f"Please fill in {field.replace('_', ' ')}."

        # Selectbox checks
        if form_state.get("data_type") == "Select...":
            errors["data_type"] = "Please select the type of data you need."
        if form_state.get("sensitivity") == "Select...":
            errors["sensitivity"] = "Please select the data sensitivity level."

        # Conditional IRB requirement
        if form_state.get("sensitivity") == "Sensitive personal data":
            if not str(form_state.get("irb_reference", "")).strip():
                errors["irb_reference"] = (
                    "IRB / ethics approval reference is required for sensitive personal data."
                )

        return errors


# ---------------------------------------------------------------------
# Mock agent wrapper (replace with your LangGraph-based AR_agent)
# ---------------------------------------------------------------------


class MockARAgent:
    """
    A simple in-process agent that mimics how you'd wrap your LangGraph AR_agent.

    Responsibilities:
    - Hold current form_state and validation_errors in its internal memory.
    - Provide:
        - sync_form(form_state) -> validation_errors
        - chat(message) -> {reply, form_updates, validation_errors}
    """

    def __init__(self, schema: FormSchema) -> None:
        self.schema = schema
        self.form_state: Dict[str, Any] = {}
        self.validation_errors: Dict[str, str] = {}
        self.history: List[Dict[str, str]] = []

    # --- API you'd call from Streamlit ---

    def sync_form(self, form_state: Dict[str, Any]) -> Dict[str, str]:
        """
        Update internal form_state and re-run validation.
        No chat message is generated here.
        """
        self.form_state = form_state.copy()
        self.validation_errors = self.schema.validate(self.form_state)
        return self.validation_errors

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Process a user message, possibly update the form, and return:
        {
          "reply": str,
          "form_updates": Dict[str, Any],
          "validation_errors": Dict[str, str],
        }
        """
        self.history.append({"role": "user", "content": message})
        msg_lower = message.lower().strip()

        # 1) Check for "set ..." style commands and update form
        form_updates = self._apply_set_commands(message)

        # 2) Refresh validation after any updates
        if form_updates:
            self.validation_errors = self.schema.validate(self.form_state)

        # 3) Handle "review" / "check" style messages
        if "review" in msg_lower or "check" in msg_lower or "ready to submit" in msg_lower:
            reply = self._review_reply()
        # 4) Provide basic explanations
        elif "justification" in msg_lower and ("why" in msg_lower or "what" in msg_lower):
            reply = (
                "The justification explains why you need this data and how it will be used. "
                "Approvers use it to assess necessity and proportionality. "
                "Be specific about goals and how you will safeguard the data."
            )
        elif ("irb" in msg_lower or "ethics" in msg_lower) and ("why" in msg_lower or "what" in msg_lower):
            reply = (
                "An IRB / Ethics approval reference is required for sensitive personal data. "
                "It shows that an independent committee has reviewed and approved the protocol."
            )
        # 5) Default: show helpful context + suggestions
        else:
            reply = self._default_reply()

        self.history.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "form_updates": form_updates,
            "validation_errors": self.validation_errors,
        }

    # --- Internal helpers ---

    def _apply_set_commands(self, message: str) -> Dict[str, Any]:
        """
        Very simple parser for commands like:
            - "set project name to ..."
            - "set data type to genomic data"
        This is purely for demo; your real LangGraph agent can be more sophisticated.
        """
        text = message.lower()
        updates: Dict[str, Any] = {}

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

                # Normalize some selectbox values
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

                self.form_state[key] = value
                updates[key] = value

        return updates

    def _review_reply(self) -> str:
        if not self.validation_errors:
            return (
                "I've checked your request and everything looks complete. "
                "You can submit it now, and I can initiate the approval workflow."
            )
        else:
            lines = [
                "Here’s what still needs attention before you submit:",
                "",
            ]
            for field, msg in self.validation_errors.items():
                label = field.replace("_", " ").title()
                lines.append(f"- {label}: {msg}")
            lines.append("")
            lines.append("Once you fix these, ask me to review again.")
            return "\n".join(lines)

    def _default_reply(self) -> str:
        """
        Default response describing what the agent can do and showing a brief snapshot.
        """
        snapshot_lines = [
            f"Project name: {self.form_state.get('project_name') or '—'}",
            f"Requester: {self.form_state.get('requester_name') or '—'}",
            f"Data type: {self.form_state.get('data_type', '—')}",
            f"Sensitivity: {self.form_state.get('sensitivity', '—')}",
            f"Duration: {self.form_state.get('duration', '—')}",
        ]
        if self.form_state.get("sensitivity") == "Sensitive personal data":
            snapshot_lines.append(f"IRB reference: {self.form_state.get('irb_reference') or '—'}")
        snapshot_lines.append(
            "Justification: " + ("present" if self.form_state.get("justification", "").strip() else "missing")
        )
        snapshot = "\n".join(snapshot_lines)

        return (
            "I’m tracking your form as you edit it. You can:\n"
            "- Ask me to review your request\n"
            "- Ask for help choosing a sensitivity level or duration\n"
            "- Ask me to draft or refine your justification\n"
            "- Update fields via chat (e.g. “set project name to Phoenix”)\n\n"
            "Here’s your current snapshot:\n\n"
            f"{snapshot}"
        )


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------


def init_state() -> None:
    """Initialize Streamlit session state keys, including the agent."""
    defaults = {
        "project_name": "",
        "requester_name": "",
        "data_type": "Select...",
        "sensitivity": "Select...",
        "justification": "",
        "duration": "3 months",
        "irb_reference": "",
        "chat_history": [],  # list of {"role": "...", "content": "..."}
        "validation_errors": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Create a single agent instance and keep it in session_state
    if "agent" not in st.session_state:
        schema = FormSchema()
        st.session_state.agent = MockARAgent(schema=schema)


def compute_progress(errors: Dict[str, str]) -> Tuple[float, str]:
    """Completion indicator based on required fields."""
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
            if value.strip():
                filled += 1
        else:
            if value:
                filled += 1

    total = len(required_fields)
    ratio = filled / total if total > 0 else 0.0
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


def render_chat(errors: Dict[str, str]) -> None:
    st.markdown("### Assistant")
    st.caption("Ask questions, get explanations, or tell me to update the form for you.")

    # Status card
    with st.container():
        completion_pct = int(compute_progress(errors)[0] * 100)
        st.markdown(
            "<div style='padding:0.75rem 1rem; border-radius:0.75rem; "
            "background-color:#EDF2F7; font-size:0.85rem;'>"
            f"<b>Current form status</b><br/>Completion: {completion_pct}%<br/>"
            f"Open issues: {len(errors)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_message = st.chat_input("Type your question or instruction...")
    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Pass message to the agent (which knows the current form via sync_form)
        result = st.session_state.agent.chat(user_message)

        # Apply form updates returned by the agent
        for key, value in result.get("form_updates", {}).items():
            if key in st.session_state:
                st.session_state[key] = value

        # Update errors from the agent
        st.session_state.validation_errors = result.get("validation_errors", {})

        reply = result.get("reply", "")
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
        st.caption("You can also ask the assistant to review your request.")

    with col_actions:
        submit_disabled = bool(errors)
        if st.button("Submit request", disabled=submit_disabled, use_container_width=True):
            st.success(
                "Your request has been submitted. The approval workflow has been initiated. "
                "You will be notified once a decision is made."
            )


def main() -> None:
    st.set_page_config(
        page_title="Data Access Request – Form & Agent",
        layout="wide",
    )
    init_state()

    # 1. Build current form_state dict from session_state
    form_state = {
        "project_name": st.session_state.project_name,
        "requester_name": st.session_state.requester_name,
        "data_type": st.session_state.data_type,
        "sensitivity": st.session_state.sensitivity,
        "duration": st.session_state.duration,
        "irb_reference": st.session_state.irb_reference,
        "justification": st.session_state.justification,
    }

    # 2. Notify the agent about the latest form values and get validation errors
    current_errors = st.session_state.agent.sync_form(form_state)
    st.session_state.validation_errors = current_errors

    # 3. Layout: form (left) + chat (right)
    col_form, col_chat = st.columns([2, 1], gap="large")

    with col_form:
        render_form(current_errors)

    with col_chat:
        render_chat(current_errors)

    render_footer(current_errors)


if __name__ == "__main__":
    main()
