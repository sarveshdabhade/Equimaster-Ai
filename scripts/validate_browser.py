"""Browser acceptance against a running local dashboard; writes review evidence."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright
from src.artifacts import atomic_json
from src.artifacts import resolve_artifact
from config import ALL_DOWNLOAD_TICKERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    output = ROOT / "logs/browser_exports"
    output.mkdir(parents=True, exist_ok=True)
    screenshots = ROOT / "images/release"
    screenshots.mkdir(exist_ok=True)
    checks, errors, console_errors, http_errors = [], [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("response", lambda response: http_errors.append({"url": response.url, "status": response.status}) if response.status >= 400 else None)

        def check(name, action):
            try:
                detail = action()
                exceptions = page.locator('[data-testid="stException"]')
                if exceptions.count():
                    raise AssertionError(exceptions.first.inner_text())
                checks.append({"name": name, "passed": True, **({"detail": detail} if detail else {})})
            except Exception as exc:
                checks.append({"name": name, "passed": False, "error": str(exc)[:1500]})
            atomic_json(ROOT / "data/evaluation/browser_validation.json", {
                "as_of": datetime.now(timezone.utc).isoformat(), "base_url": base, "checks": checks,
                "page_errors": errors, "console_errors": console_errors, "http_errors": http_errors})
            print(json.dumps(checks[-1]), flush=True)

        def download(name):
            button = page.get_by_role("button", name=name, exact=True)
            button.wait_for(timeout=90000)
            with page.expect_download(timeout=30000) as pending:
                button.click()
            item = pending.value
            target = output / item.suggested_filename
            item.save_as(target)
            assert target.stat().st_size > 10

        def home():
            page.goto(base, wait_until="domcontentloaded")
            page.get_by_text("Markets", exact=True).last.wait_for(timeout=90000)
            page.get_by_text("Markets", exact=True).last.click()
            page.get_by_text("Coverage breadth", exact=False).first.wait_for(timeout=90000)
            assert "58 stocks" in page.locator("body").inner_text()
            page.screenshot(path=str(screenshots / "home_desktop.png"), full_page=True)

        def screener():
            page.get_by_text("Stock screener", exact=True).click()
            download("Export screener")

        def overview():
            page.goto(base + "/stock_analyzer?ticker=TCS.NS", wait_until="domcontentloaded")
            download("Export price history")

        def technical():
            page.get_by_text("Technical Chart", exact=True).click()
            page.get_by_text("Chart date range", exact=True).wait_for(timeout=90000)
            page.locator('[data-testid="stPlotlyChart"]').first.wait_for(timeout=90000)

        def forecast():
            page.get_by_text("LSTM Model", exact=True).click()
            page.get_by_role("button", name="Generate Final Estimate", exact=True).wait_for(timeout=90000)
            page.get_by_role("button", name="Generate Final Estimate", exact=True).click()
            download("Export Forecast CSV")
            assert "LSTM baseline" in page.locator("body").inner_text()
            page.screenshot(path=str(screenshots / "forecast_tcs.png"), full_page=True)

        def news():
            page.get_by_text("News Sentiment", exact=True).click()
            download("Export news evidence")

        def fundamentals():
            page.get_by_text("Fundamentals", exact=True).click()
            download("Export All Fundamentals")
            page.get_by_text("Statements", exact=True).click()
            download("Export Quarterly results")
            page.get_by_text("Peer Comparison", exact=True).click()
            download("Export Peer Comparison")

        def unavailable():
            missing = None
            for ticker in [*ALL_DOWNLOAD_TICKERS, "NSEI"]:
                try:
                    resolve_artifact(ticker)
                except FileNotFoundError:
                    missing = ticker
                    break
            if missing:
                page.goto(base + f"/stock_analyzer?ticker={missing}", wait_until="domcontentloaded")
                page.get_by_text("LSTM Model", exact=True).wait_for(timeout=90000)
                page.get_by_text("LSTM Model", exact=True).click()
                page.get_by_text("LSTM Brain offline. Model files not found.", exact=True).wait_for(timeout=90000)
                return "Missing-model warning exercised for " + missing
            page.goto(base + "/stock_analyzer?ticker=OUTSIDE-COVERAGE", wait_until="domcontentloaded")
            page.get_by_text("That symbol is outside the current research coverage. Select a company below.", exact=True).wait_for(timeout=90000)
            return "All models available; unsupported-symbol warning exercised. Missing-model rendering also has offline regression coverage."

        def mobile():
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(base, wait_until="domcontentloaded")
            # A sidebar opened at desktop width can retain its state after a
            # resize; exercise the user's collapse control before navigation.
            collapse = page.locator('[data-testid="stSidebarCollapseButton"] button')
            if collapse.count() and collapse.first.is_visible():
                collapse.first.click()
            page.get_by_text("Markets", exact=True).last.wait_for(timeout=90000)
            page.get_by_text("Markets", exact=True).last.click()
            page.get_by_text("Coverage breadth", exact=False).first.wait_for(timeout=90000)
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            page.screenshot(path=str(screenshots / "home_mobile.png"), full_page=True)

        for name, action in (("Home desktop", home), ("Screener CSV", screener),
                             ("Company overview and price CSV", overview), ("Technical chart", technical),
                             ("Forecast and CSV", forecast), ("News evidence CSV", news),
                             ("Fundamentals, statements and peers CSV", fundamentals),
                             ("Unavailable symbol/model state", unavailable), ("Mobile layout", mobile)):
            check(name, action)
        browser.close()
    blocking_http = [entry for entry in http_errors if not (
        entry["status"] == 404 and entry["url"].endswith((
            "/stock_analyzer/_stcore/health", "/stock_analyzer/_stcore/host-config")))]
    atomic_json(ROOT / "data/evaluation/browser_validation.json", {
        "as_of": datetime.now(timezone.utc).isoformat(), "base_url": base, "checks": checks,
        "page_errors": errors, "console_errors": console_errors,
        "http_errors": http_errors, "blocking_http_errors": blocking_http,
        "framework_probe_note": "Direct multipage navigation probes a route-relative Streamlit health/config URL before using the root endpoint."})
    return int(any(not check["passed"] for check in checks) or bool(errors) or bool(blocking_http))


if __name__ == "__main__":
    raise SystemExit(main())
