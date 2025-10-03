"""
Drop‑in function to handle both JavaScript alerts/confirmations and HTML modal dialogs
(Bootstrap, Material‑UI, Ant Design, Chakra, etc.).

Usage (within your scraper after triggering a download or a risky action):

    result = handle_popups(
        driver,
        action="accept",              # or "dismiss"
        timeout=15,
        config=ModalConfig(
            # If you know exact confirm/cancel button selectors, set them:
            # confirm_selector="#confirm-download",
            # cancel_selector="#cancel",
            # Otherwise heuristics will match by button text and common CSS classes.
        ),
    )
    logger.info("Popup handled: %s", result)

The function returns a dict, e.g.:
    {"type": "modal", "action": "accept", "text": "Ready to download?", "button_text": "Download"}

Integrates cleanly with the retry strategy from the main script; you can wrap
`handle_popups` with the same @retry decorator if your site sometimes opens, closes,
then reopens modals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException


@dataclass
class ModalConfig:
    # Broad set of modal container selectors across popular UI kits
    modal_selectors: List[str] = field(default_factory=lambda: [
        "[role='dialog'][aria-modal='true']",             # ARIA spec
        ".modal.show, .modal[style*='display: block']",  # Bootstrap
        ".MuiDialog-root[aria-hidden='false']",          # Material‑UI
        ".ant-modal-wrap .ant-modal",                    # Ant Design
        ".chakra-modal__content-container",              # Chakra UI
        "[data-testid*='modal']",                        # Testing hooks
        "[class*='Dialog'][class*='open']",              # generic fallbacks
    ])

    # Backdrop/overlay selectors to wait for invisibility after closing
    backdrop_selectors: List[str] = field(default_factory=lambda: [
        ".modal-backdrop.show", ".MuiBackdrop-root", ".ant-modal-mask", ".chakra-modal__overlay"
    ])

    # Optional explicit selectors for confirm/cancel buttons (recommended when stable)
    confirm_selector: Optional[str] = None
    cancel_selector: Optional[str] = None

    # Text heuristics if explicit selectors are not provided (case‑insensitive)
    confirm_texts: List[str] = field(default_factory=lambda: [
        "ok", "yes", "confirm", "continue", "download", "accept", "proceed", "submit", "save", "agree"
    ])
    cancel_texts: List[str] = field(default_factory=lambda: [
        "cancel", "no", "close", "dismiss", "x"
    ])

    # Button selectors to search inside a modal
    button_selectors: List[str] = field(default_factory=lambda: [
        "button", "[role='button']", ".btn", ".ant-btn", ".MuiButton-root"
    ])

    # Primary/confirm visual classes used as a last resort
    primary_button_hints: List[str] = field(default_factory=lambda: [
        ".btn-primary", ".ant-btn-primary", ".MuiButton-containedPrimary"
    ])


def _text(el) -> str:
    try:
        return (el.text or "").strip()
    except StaleElementReferenceException:
        return ""


def _visible(el) -> bool:
    try:
        return el.is_displayed()
    except StaleElementReferenceException:
        return False


def _find_first(driver, modal, selectors: List[str]):
    for sel in selectors:
        try:
            els = modal.find_elements(By.CSS_SELECTOR, sel)
        except StaleElementReferenceException:
            return None
        for el in els:
            if _visible(el):
                return el
    return None


def _find_buttons(driver, modal, selectors: List[str]):
    buttons = []
    for sel in selectors:
        try:
            buttons.extend(modal.find_elements(By.CSS_SELECTOR, sel))
        except StaleElementReferenceException:
            return []
    return [b for b in buttons if _visible(b)]


def _match_button_by_text(buttons, keywords: List[str]):
    keys = {k.lower() for k in keywords}
    for b in buttons:
        label = _text(b).lower()
        if any(k == label or (len(k) >= 2 and k in label) for k in keys):
            return b
    return None


def _wait_modal_closed(driver, modal, backdrops: List[str], timeout: int = 10) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        # Modal gone or invisible
        try:
            if not _visible(modal):
                # Also ensure no backdrops remain visible
                if not any(any(e.is_displayed() for e in driver.find_elements(By.CSS_SELECTOR, sel)) for sel in backdrops):
                    return True
        except StaleElementReferenceException:
            # Detached from DOM → closed
            return True
        time.sleep(0.1)
    return False


def _extract_modal_text(modal) -> str:
    # Prefer body/content areas when present
    preferred = [
        ".modal-body", ".ant-modal-body", ".MuiDialogContent-root", ".chakra-modal__content",
        "[data-testid*='modal-body']", "[role='document']"
    ]
    for sel in preferred:
        try:
            els = modal.find_elements(By.CSS_SELECTOR, sel)
        except StaleElementReferenceException:
            break
        for el in els:
            if _visible(el):
                txt = _text(el)
                if txt:
                    return txt
    return _text(modal)


def handle_popups(driver, action: str = "accept", timeout: int = 10, config: Optional[ModalConfig] = None) -> Dict[str, str]:
    """Handle either a JS alert/confirm or an HTML modal dialog.

    Parameters
    ----------
    driver : WebDriver
        Selenium driver instance.
    action : {"accept", "dismiss"}
        What to do if a popup is found. For modals, attempts to click an appropriate button.
    timeout : int
        Max seconds to wait for a popup to appear.
    config : ModalConfig
        Configuration for selectors and heuristics.

    Returns
    -------
    dict
        Details of the handled popup: type ("alert"|"modal"), action taken, text, and optional button_text.

    Raises
    ------
    TimeoutException
        If no popup appears within timeout, or modal has no suitable button.
    """
    config = config or ModalConfig()

    # Phase 1: Try JavaScript alerts/confirmations first.
    try:
        WebDriverWait(driver, timeout).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        text = alert.text
        if action == "dismiss":
            alert.dismiss()
        else:
            alert.accept()
        return {"type": "alert", "action": action, "text": text}
    except TimeoutException:
        pass

    # Phase 2: Look for visible HTML modals.
    end = time.time() + timeout
    last_exc = None
    while time.time() < end:
        visible_modals = []
        for sel in config.modal_selectors:
            try:
                for el in driver.find_elements(By.CSS_SELECTOR, sel):
                    if _visible(el):
                        visible_modals.append(el)
            except StaleElementReferenceException:
                continue
        if not visible_modals:
            time.sleep(0.1)
            continue

        # Heuristic: pick the last visible (often topmost)
        modal = visible_modals[-1]
        modal_text = _extract_modal_text(modal)

        # Choose confirm/cancel button
        target_button = None
        if action == "accept" and config.confirm_selector:
            target_button = _find_first(driver, modal, [config.confirm_selector])
        elif action == "dismiss" and config.cancel_selector:
            target_button = _find_first(driver, modal, [config.cancel_selector])

        if target_button is None:
            # Fallback: search all buttons and then by label; last resort: primary class hints
            buttons = _find_buttons(driver, modal, config.button_selectors)
            if action == "accept":
                target_button = _match_button_by_text(buttons, config.confirm_texts)
                if target_button is None:
                    target_button = _find_first(driver, modal, config.primary_button_hints)
            else:
                target_button = _match_button_by_text(buttons, config.cancel_texts)
                if target_button is None:
                    # Common close buttons: data-dismiss or top-right X
                    for b in buttons:
                        attrs = ["data-dismiss", "aria-label"]
                        try:
                            if any((b.get_attribute(a) or "").lower() in ("modal", "close") for a in attrs):
                                target_button = b
                                break
                        except StaleElementReferenceException:
                            continue

        if target_button is None:
            last_exc = NoSuchElementException("No suitable button found in modal")
            # Try the next iteration (modal might still be initializing)
            time.sleep(0.2)
            continue

        # Click with a small wait; fallback to JS click if needed
        try:
            try:
                WebDriverWait(driver, 5).until(lambda d: target_button.is_enabled() and _visible(target_button))
                target_button.click()
            except Exception:
                driver.execute_script("arguments[0].click();", target_button)
        except Exception as e:
            last_exc = e
            time.sleep(0.2)
            continue

        # Wait for modal to fully close (and backdrop disappear)
        if not _wait_modal_closed(driver, modal, config.backdrop_selectors, timeout=timeout):
            # Sometimes clicking triggers async work; we still return if the button was clicked.
            # If you require strict closure, raise here instead.
            pass

        return {
            "type": "modal",
            "action": action,
            "text": modal_text,
            "button_text": _text(target_button),
        }

    raise TimeoutException("No alert or modal appeared within the timeout, or interaction failed") from last_exc
