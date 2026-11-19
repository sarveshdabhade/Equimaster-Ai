"""Capture a public or local UI using the installed Edge browser."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--full-page", action="store_true")
    parser.add_argument("--wait-for", help="Visible text indicating the Streamlit view has rendered")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        if args.wait_for:
            page.get_by_text(args.wait_for, exact=False).first.wait_for(state="visible", timeout=90000)
        page.wait_for_timeout(6000)
        page.screenshot(path=args.output, full_page=args.full_page)
        print({"title": page.title(), "url": page.url, "output": args.output,
               "horizontal_overflow": page.evaluate("document.documentElement.scrollWidth > innerWidth")})
        browser.close()


if __name__ == "__main__":
    main()
