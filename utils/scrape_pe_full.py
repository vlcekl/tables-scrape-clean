"""
Unified, robust Selenium downloader that:
  • Retries brittle actions with exponential backoff + jitter
  • Uses only explicit waits (no implicit waits)
  • Handles BOTH JS alerts AND HTML modals (Bootstrap/MUI/Ant/Chakra/etc.)
  • Verifies the downloaded CSV (header + at least one data row)

Run (toy example):
  python scraper_modal_retry.py \
      --time-range "Last 30 Days" \
      --data-type "Orders" \
      --expected-header "order_id,sku,qty,price" \
      --popup-action accept \
      --popup-timeout 15 \
      --confirm-selector ".modal-footer .btn-primary"

Replace selectors and URL with your real internal site.
"""

from __future__ import annotations

import os
import csv
import glob
import time
import random
import string
import shutil
import logging
import argparse
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Tuple, Type, List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
    NoSuchElementException,
    WebDriverException,
)

# ----------------------- Logging -------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------- Config & Selectors --------------------------------
@dataclass
class Selectors:
    url: str = "https://example.com/data"  # TODO: replace with internal page
    time_selector: str = "#time-range"      # e.g., <select id="time-range"> ...
    type_selector: str = "#data-type"       # e.g., <select id="data-type"> ...
    download_button: str = "#download"      # e.g., <button id="download">Download</button>

# ----------------------- Retry Decorator -----------------------------------
class VerificationError(Exception):
    """Raised when post-conditions (e.g., file content) do not verify."""

def retry(
    exceptions: Tuple[Type[BaseException], ...] = (
        TimeoutException,
        StaleElementReferenceException,
        ElementClickInterceptedException,
        NoSuchElementException,
        WebDriverException,
    ),
    tries: int = 4,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.3,
    on_retry: Optional[Callable[[BaseException, int], None]] = None,
):
    """Retry calling the decorated function using an exponential backoff."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exc: Optional[BaseException] = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if on_retry:
                        try:
                            on_retry(exc, attempt)
                        except Exception:
                            pass
                    if attempt == tries:
                        break
                    sleep_time = delay * (backoff ** (attempt - 1)) + random.uniform(0, jitter)
                    logger.warning(
                        "Attempt %s/%s failed due to %s. Retrying in %.2fs...",
                        attempt, tries, type(exc).__name__, sleep_time
                    )
                    time.sleep(sleep_time)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator

# ----------------------- Driver Setup --------------------------------------

def make_chrome(download_dir: str, headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(0)  # Prefer explicit waits only
    return driver

# ----------------------- Wait & Action Helpers -----------------------------

def wait_visible(driver: webdriver.Chrome, selector: str, by: By = By.CSS_SELECTOR, timeout: int = 20):
    return WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((by, selector)))

def wait_clickable(driver: webdriver.Chrome, selector: str, by: By = By.CSS_SELECTOR, timeout: int = 20):
    return WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((by, selector)))

def _js_click(driver: webdriver.Chrome, el):
    driver.execute_script("arguments[0].click();", el)

@retry()
def safe_click(driver: webdriver.Chrome, selector: str, by: By = By.CSS_SELECTOR, timeout: int = 20) -> None:
    el = wait_clickable(driver, selector, by=by, timeout=timeout)
    try:
        el.click()
    except (ElementClickInterceptedException, WebDriverException):
        _js_click(driver, el)

@retry()
def safe_select_by_text(
    driver: webdriver.Chrome,
    selector: str,
    visible_text: str,
    by: By = By.CSS_SELECTOR,
    timeout: int = 20,
) -> None:
    select_el = wait_visible(driver, selector, by=by, timeout=timeout)
    Select(select_el).select_by_visible_text(visible_text)

# ----------------------- Modal & Alert Handling ----------------------------
@dataclass
class ModalConfig:
    modal_selectors: List[str] = field(default_factory=lambda: [
        "[role='dialog'][aria-modal='true']",
        ".modal.show, .modal[style*='display: block']",
        ".MuiDialog-root[aria-hidden='false']",
        ".ant-modal-wrap .ant-modal",
        ".chakra-modal__content-container",
        "[data-testid*='modal']",
        "[class*='Dialog'][class*='open']",
    ])
    backdrop_selectors: List[str] = field(default_factory=lambda: [
        ".modal-backdrop.show", ".MuiBackdrop-root", ".ant-modal-mask", ".chakra-modal__overlay"
    ])
    confirm_selector: Optional[str] = None
    cancel_selector: Optional[str] = None
    confirm_texts: List[str] = field(default_factory=lambda: [
        "ok", "yes", "confirm", "continue", "download", "accept", "proceed", "submit", "save", "agree"
    ])
    cancel_texts: List[str] = field(default_factory=lambda: [
        "cancel", "no", "close", "dismiss", "x"
    ])
    button_selectors: List[str] = field(default_factory=lambda: [
        "button", "[role='button']", ".btn", ".ant-btn", ".MuiButton-root"
    ])
    primary_button_hints: List[str] = field(default_factory=lambda: [
        ".btn-primary", ".ant-btn-primary", ".MuiButton-containedPrimary"
    ])

def _el_text(el) -> str:
    try:
        return (el.text or "").strip()
    except StaleElementReferenceException:
        return ""

def _el_visible(el) -> bool:
    try:
        return el.is_displayed()
    except StaleElementReferenceException:
        return False

def _find_first_in_modal(modal, selectors: List[str]):
    for sel in selectors:
        try:
            els = modal.find_elements(By.CSS_SELECTOR, sel)
        except StaleElementReferenceException:
            return None
        for el in els:
            if _el_visible(el):
                return el
    return None

def _find_buttons_in_modal(modal, selectors: List[str]):
    buttons = []
    for sel in selectors:
        try:
            buttons.extend(modal.find_elements(By.CSS_SELECTOR, sel))
        except StaleElementReferenceException:
            return []
    return [b for b in buttons if _el_visible(b)]

def _match_button_by_text(buttons, keywords: List[str]):
    keys = {k.lower() for k in keywords}
    for b in buttons:
        label = _el_text(b).lower()
        if any(k == label or (len(k) >= 2 and k in label) for k in keys):
            return b
    return None

def _wait_modal_closed(driver, modal, backdrops: List[str], timeout: int = 10) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if not _el_visible(modal):
                # verify no backdrops visible
                if not any(any(e.is_displayed() for e in driver.find_elements(By.CSS_SELECTOR, sel)) for sel in backdrops):
                    return True
        except StaleElementReferenceException:
            return True
        time.sleep(0.1)
    return False

def _extract_modal_text(modal) -> str:
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
            if _el_visible(el):
                txt = _el_text(el)
                if txt:
                    return txt
    return _el_text(modal)

@retry()
def handle_popups(driver, action: str = "accept", timeout: int = 10, config: Optional[ModalConfig] = None) -> Dict[str, str]:
    """Handle a JS alert/confirm or an HTML modal dialog; returns details dict."""
    config = config or ModalConfig()

    # Try alerts first
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

    # Try visible HTML modals
    end = time.time() + timeout
    while time.time() < end:
        visible_modals = []
        for sel in config.modal_selectors:
            try:
                visible_modals.extend([
                    el for el in driver.find_elements(By.CSS_SELECTOR, sel) if _el_visible(el)
                ])
            except StaleElementReferenceException:
                continue
        if not visible_modals:
            time.sleep(0.1)
            continue

        modal = visible_modals[-1]  # heuristic: top-most
        modal_text = _extract_modal_text(modal)

        target_button = None
        if action == "accept" and config.confirm_selector:
            target_button = _find_first_in_modal(modal, [config.confirm_selector])
        elif action == "dismiss" and config.cancel_selector:
            target_button = _find_first_in_modal(modal, [config.cancel_selector])

        if target_button is None:
            buttons = _find_buttons_in_modal(modal, config.button_selectors)
            if action == "accept":
                target_button = _match_button_by_text(buttons, config.confirm_texts)
                if target_button is None:
                    target_button = _find_first_in_modal(modal, config.primary_button_hints)
            else:
                target_button = _match_button_by_text(buttons, config.cancel_texts)
                if target_button is None:
                    # common close buttons
                    for b in buttons:
                        try:
                            attrs = ["data-dismiss", "aria-label"]
                            if any((b.get_attribute(a) or "").lower() in ("modal", "close") for a in attrs):
                                target_button = b
                                break
                        except StaleElementReferenceException:
                            continue

        if target_button is None:
            time.sleep(0.2)
            continue

        try:
            try:
                WebDriverWait(driver, 5).until(lambda d: target_button.is_enabled() and _el_visible(target_button))
                target_button.click()
            except Exception:
                driver.execute_script("arguments[0].click();", target_button)
        except Exception:
            time.sleep(0.2)
            continue

        _wait_modal_closed(driver, modal, config.backdrop_selectors, timeout=timeout)
        return {"type": "modal", "action": action, "text": modal_text, "button_text": _el_text(target_button)}

    raise TimeoutException("No alert or modal appeared within the timeout")

# ----------------------- Download Monitoring & Verification ----------------

def _stable_file(path: str, interval: float = 0.5) -> bool:
    try:
        size1 = os.path.getsize(path)
        time.sleep(interval)
        size2 = os.path.getsize(path)
        return size1 == size2
    except FileNotFoundError:
        return False

def wait_for_download_complete(
    download_dir: str,
    startswith: Optional[str] = None,
    endswith: str = ".csv",
    timeout: int = 180,
    min_size: int = 50,
) -> str:
    logger.info("Waiting for download to complete in %s ...", download_dir)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if glob.glob(os.path.join(download_dir, "*.crdownload")):
            time.sleep(0.3)
            continue
        candidates = [
            p for p in glob.glob(os.path.join(download_dir, f"*{endswith}"))
            if os.path.getsize(p) >= min_size
        ]
        if startswith:
            candidates = [p for p in candidates if os.path.basename(p).startswith(startswith)]
        if candidates:
            latest = max(candidates, key=os.path.getmtime)
            if _stable_file(latest):
                logger.info("Download detected: %s", latest)
                return latest
        time.sleep(0.3)
    raise TimeoutException("Timed out waiting for CSV download to complete.")

def verify_csv(path: str, expected_header: Optional[Sequence[str]] = None, require_rows: bool = True) -> None:
    logger.info("Verifying CSV: %s", path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if expected_header is not None and list(header or []) != list(expected_header):
            raise VerificationError(f"Header mismatch: got {header}, expected {list(expected_header)}")
        if require_rows:
            first_row = next(reader, None)
            if first_row is None:
                raise VerificationError("CSV contains no data rows.")

# ----------------------- High-level Steps (Toy Flow) -----------------------

def _make_on_retry(driver: webdriver.Chrome, download_dir: str) -> Callable[[BaseException, int], None]:
    def _hook(exc: BaseException, attempt: int) -> None:
        # Screenshot per attempt
        fname = os.path.join(download_dir, f"attempt_{attempt}_error.png")
        try:
            driver.save_screenshot(fname)
            logger.warning("Saved screenshot: %s", fname)
        except Exception:
            pass
        if attempt >= 2:
            try:
                driver.refresh()
                logger.info("Refreshed page after failure (%s).", type(exc).__name__)
            except Exception:
                pass
    return _hook

def sanitize(s: str) -> str:
    keep = string.ascii_letters + string.digits + "-_."
    return "".join(ch if ch in keep else "_" for ch in s)[:80]

@retry()
def go_to_page(driver: webdriver.Chrome, selectors: Selectors) -> None:
    logger.info("Navigating to %s", selectors.url)
    driver.get(selectors.url)
    wait_visible(driver, selectors.time_selector)

@retry()
def set_filters(driver: webdriver.Chrome, selectors: Selectors, time_range: str, data_type: str) -> None:
    safe_select_by_text(driver, selectors.time_selector, time_range)
    safe_select_by_text(driver, selectors.type_selector, data_type)

@retry()
def click_download(driver: webdriver.Chrome, selectors: Selectors) -> None:
    safe_click(driver, selectors.download_button)

# ----------------------- Orchestrator --------------------------------------

def run_scrape(
    time_range: str,
    data_type: str,
    expected_header: Optional[Sequence[str]] = None,
    selectors: Optional[Selectors] = None,
    headless: bool = True,
    popup_action: str = "accept",
    popup_timeout: int = 15,
    modal_config: Optional[ModalConfig] = None,
) -> str:
    selectors = selectors or Selectors()
    download_dir = tempfile.mkdtemp(prefix="selenium_dl_")
    logger.info("Download directory: %s", download_dir)

    driver = make_chrome(download_dir, headless=headless)

    # Strengthen key phases with diagnostics on retry
    go_to = retry(on_retry=_make_on_retry(driver, download_dir))(go_to_page)
    set_filt = retry(on_retry=_make_on_retry(driver, download_dir))(set_filters)
    click_dl = retry(on_retry=_make_on_retry(driver, download_dir))(click_download)
    handle_pop = retry(on_retry=_make_on_retry(driver, download_dir))(handle_popups)

    try:
        # 1) Navigate & ensure ready
        go_to(driver, selectors)

        # 2) Apply filters
        set_filt(driver, selectors, time_range, data_type)

        # 3) Trigger download
        click_dl(driver, selectors)

        # 4) Handle any popup (alert or modal)
        try:
            result = handle_pop(driver, action=popup_action, timeout=popup_timeout, config=modal_config)
            logger.info("Popup handled: %s", result)
        except TimeoutException:
            logger.info("No alert or modal appeared within %ss; continuing...", popup_timeout)

        # 5) Wait for and verify the CSV
        csv_path = wait_for_download_complete(download_dir, startswith=None, endswith=".csv", timeout=180)
        verify_csv(csv_path, expected_header=expected_header, require_rows=True)

        dest = os.path.join(
            download_dir, f"download_{sanitize(time_range)}_{sanitize(data_type)}.csv"
        )
        shutil.copy2(csv_path, dest)
        logger.info("Downloaded & verified: %s", dest)
        return dest

    finally:
        driver.quit()

# ----------------------- CLI ----------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy Selenium downloader with robust retry and modal/alert handling")
    p.add_argument("--time-range", required=True, help="Visible text of the time range option")
    p.add_argument("--data-type", required=True, help="Visible text of the data type option")
    p.add_argument("--expected-header", default=None, help="Comma-separated header to verify (e.g., 'col1,col2')")
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode (default: true)")

    # Popup handling configuration
    p.add_argument("--popup-action", choices=["accept", "dismiss"], default="accept")
    p.add_argument("--popup-timeout", type=int, default=15)
    p.add_argument("--confirm-selector", default=None, help="CSS selector for modal confirm button (optional)")
    p.add_argument("--cancel-selector", default=None, help="CSS selector for modal cancel/close button (optional)")
    p.add_argument("--confirm-texts", default=None, help="Comma-separated confirm keywords (fallback)")
    p.add_argument("--cancel-texts", default=None, help="Comma-separated cancel keywords (fallback)")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    expected = [h.strip() for h in args.expected_header.split(",")] if args.expected_header else None

    # Build modal configuration (example for Bootstrap/MUI‑like modals)
    modal_cfg = ModalConfig(
        confirm_selector=args.confirm_selector or ".modal-footer .btn-primary",  # example default
        cancel_selector=args.cancel_selector,                                     # leave None if not needed
    )
    if args.confirm_texts:
        modal_cfg.confirm_texts = [t.strip() for t in args.confirm_texts.split(",") if t.strip()]
    if args.cancel_texts:
        modal_cfg.cancel_texts = [t.strip() for t in args.cancel_texts.split(",") if t.strip()]

    path = run_scrape(
        time_range=args.time_range,
        data_type=args.data_type,
        expected_header=expected,
        selectors=Selectors(),
        headless=True,  # keep headless by default for CI; toggle if you need to see the browser
        popup_action=args.popup_action,
        popup_timeout=args.popup_timeout,
        modal_config=modal_cfg,
    )
    print(path)
