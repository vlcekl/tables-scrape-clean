"""
Robust Selenium scraper skeleton with systematic retry, explicit waits,
clean abstractions, and verified CSV downloads (toy example).

Key ideas implemented here:
  1) Never store WebElement references between retries; re-locate every time.
  2) Wrap actions in a reusable @retry decorator with exponential backoff + jitter.
  3) Use explicit waits (WebDriverWait + EC.*) everywhere.
  4) Harden interactions (JS click fallback, Select with retries) and handle
     StaleElementReferenceException, TimeoutException, click interception, etc.
  5) Make steps idempotent and verifiable. We verify the downloaded CSV (header,
     non-empty) and wait until Chrome finishes the download (no .crdownload, stable size).
  6) On retry, capture a screenshot and (optionally) refresh the page.

This is a TOY EXAMPLE. Replace the selectors and URL with your real internal site.

Run:
  python scraper_retry_toy.py --time-range "Last 30 Days" --data-type "Orders" \
      --expected-header "order_id,sku,qty,price"

Notes:
  • The script sets a temporary download directory and configures Chrome to auto-download.
  • A confirmation pop-up (alert) is expected after clicking the download button; we accept it.
  • On failure, a screenshot is saved in the download directory for diagnostics.
"""

from __future__ import annotations

import os
import csv
import glob
import time
import math
import shutil
import random
import string
import logging
import argparse
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, Type, List

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
    UnexpectedAlertPresentException,
)


# ----------------------- Logging -------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
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
    """Retry calling the decorated function using an exponential backoff.

    Parameters
    ----------
    exceptions : tuple
        Exceptions that trigger a retry.
    tries : int
        Total attempts.
    delay : float
        Initial delay between retries in seconds.
    backoff : float
        Multiplier applied to delay between attempts.
    jitter : float
        Random jitter added to the delay to avoid synchronized retries.
    on_retry : callable(exc, attempt)
        Optional hook called after a failed attempt.
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exc: Optional[BaseException] = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203 - explicit is fine here
                    last_exc = exc
                    if on_retry:
                        try:
                            on_retry(exc, attempt)
                        except Exception as hook_err:  # best effort
                            logger.debug("on_retry hook failed: %s", hook_err)
                    if attempt == tries:
                        break
                    sleep_time = delay * (backoff ** (attempt - 1)) + random.uniform(0, jitter)
                    logger.warning(
                        "Attempt %s/%s failed due to %s. Retrying in %.2fs...",
                        attempt,
                        tries,
                        type(exc).__name__,
                        sleep_time,
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
        # Headless 'new' is recommended on recent Chromium
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


def accept_popup_if_present(driver: webdriver.Chrome, timeout: int = 10) -> Optional[str]:
    try:
        WebDriverWait(driver, timeout).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        text = alert.text
        alert.accept()
        logger.info("Accepted alert: %s", text)
        return text
    except TimeoutException:
        return None


# ----------------------- Download Monitoring & Verification ----------------

def _stable_file(path: str, interval: float = 0.5) -> bool:
    """Return True if file size is stable across two checks."""
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
    min_size: int = 50,  # bytes
) -> str:
    """Block until a non-temporary CSV appears in download_dir and stabilizes.

    If `startswith` is given, require the filename to begin with that prefix.
    """
    logger.info("Waiting for download to complete in %s ...", download_dir)
    t0 = time.time()
    while time.time() - t0 < timeout:
        tmp_parts = glob.glob(os.path.join(download_dir, "*.crdownload"))
        if tmp_parts:
            time.sleep(0.3)
            continue
        candidates = [
            p for p in glob.glob(os.path.join(download_dir, f"*{endswith}"))
            if os.path.getsize(p) >= min_size
        ]
        if startswith:
            candidates = [p for p in candidates if os.path.basename(p).startswith(startswith)]
        if candidates:
            # Choose most recent
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
        # Capture a screenshot for post-mortem
        fname = os.path.join(download_dir, f"attempt_{attempt}_error.png")
        try:
            driver.save_screenshot(fname)
            logger.warning("Saved screenshot: %s", fname)
        except Exception:
            pass
        # Aggressive recovery on later attempts
        if attempt >= 2:
            try:
                driver.refresh()
                logger.info("Refreshed page after failure (%s).", type(exc).__name__)
            except Exception:
                pass
    return _hook


def go_to_page(driver: webdriver.Chrome, selectors: Selectors) -> None:
    logger.info("Navigating to %s", selectors.url)
    driver.get(selectors.url)
    # Wait for a key element to ensure page is ready
    wait_visible(driver, selectors.time_selector)


@retry()
def set_filters(driver: webdriver.Chrome, selectors: Selectors, time_range: str, data_type: str) -> None:
    safe_select_by_text(driver, selectors.time_selector, time_range)
    safe_select_by_text(driver, selectors.type_selector, data_type)


@retry()
def click_download(driver: webdriver.Chrome, selectors: Selectors) -> None:
    safe_click(driver, selectors.download_button)


# ----------------------- Orchestrator --------------------------------------

def sanitize(s: str) -> str:
    keep = string.ascii_letters + string.digits + "-_."
    return "".join(ch if ch in keep else "_" for ch in s)[:80]


def run_scrape(
    time_range: str,
    data_type: str,
    expected_header: Optional[Sequence[str]] = None,
    selectors: Optional[Selectors] = None,
    headless: bool = True,
) -> str:
    selectors = selectors or Selectors()
    download_dir = tempfile.mkdtemp(prefix="selenium_dl_")
    logger.info("Download directory: %s", download_dir)

    driver = make_chrome(download_dir, headless=headless)

    # Inject stronger retry w/ on_retry hook for key phases
    go_to = retry(on_retry=_make_on_retry(driver, download_dir))(go_to_page)
    set_filt = retry(on_retry=_make_on_retry(driver, download_dir))(set_filters)
    click_dl = retry(on_retry=_make_on_retry(driver, download_dir))(click_download)

    try:
        # 1) Navigate & ensure ready
        go_to(driver, selectors)

        # 2) Apply filters (idempotent)
        set_filt(driver, selectors, time_range, data_type)

        # 3) Trigger download and accept confirmation
        click_dl(driver, selectors)
        accept_popup_if_present(driver, timeout=15)

        # 4) Wait for and verify the CSV
        prefix = None  # put a filename prefix here if the site uses one
        csv_path = wait_for_download_complete(download_dir, startswith=prefix, endswith=".csv", timeout=180)
        verify_csv(csv_path, expected_header=expected_header, require_rows=True)

        # 5) Rename to something deterministic (includes filters)
        dest = os.path.join(
            download_dir, f"download_{sanitize(time_range)}_{sanitize(data_type)}.csv"
        )
        shutil.copy2(csv_path, dest)
        logger.info("Downloaded & verified: %s", dest)
        return dest

    except Exception as e:
        # Final screenshot on fatal error
        try:
            driver.save_screenshot(os.path.join(download_dir, "fatal_error.png"))
        except Exception:
            pass
        raise
    finally:
        driver.quit()


# ----------------------- CLI ----------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy Selenium downloader with robust retry")
    p.add_argument("--time-range", required=True, help="Visible text of the time range option")
    p.add_argument("--data-type", required=True, help="Visible text of the data type option")
    p.add_argument(
        "--expected-header",
        default=None,
        help="Comma-separated header to verify (e.g., 'col1,col2,col3')",
    )
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    expected = [h.strip() for h in args.expected_header.split(",")] if args.expected_header else None
    path = run_scrape(
        time_range=args.time_range,
        data_type=args.data_type,
        expected_header=expected,
        headless=args.headless or True,
    )
    print(path)
