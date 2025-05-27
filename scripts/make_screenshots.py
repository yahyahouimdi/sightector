"""Regenerate docs/screenshots/*.png with Playwright driving the installed Edge/Chrome.

Requires the API to be running:  python -m server
Usage:                           python scripts/make_screenshots.py
(one-time setup:                 pip install playwright)

The extension popup is rendered outside Chrome's extension runtime: the
``chrome.*`` APIs it uses are replaced by a tiny stub that returns a fixed
"selected text"; the script then clicks *Scan* like a user would.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
API = "http://127.0.0.1:5000"
POPUP_PORT = 8765  # must be an origin the API's CORS policy accepts

DASHBOARD_SHOTS = {
    "dashboard_obfuscated_insult.png": "you are a fu*ck iiidiot",
    "dashboard_spaced_letters.png": "f u c k you, u r a m0r0n",
    "dashboard_clean_message.png": "kill the process and restart the server",
    "dashboard_threat.png": "i will kill you",
}
POPUP_SHOTS = {
    "extension_harassment.png": "STUPIIIID b!tch, nobody wants you here",
    "extension_clean.png": "Thanks for the help with my report, see you tomorrow!",
}
CHROME_STUB = """
window.chrome = {
  runtime: { sendMessage: (_m, cb) => cb({ selectedText: %s }) },
  tabs: { query: (_q, cb) => cb([{ url: "https://www.example.com/forum/thread/42" }]) },
};
"""


def serve(directory: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                             "--directory", str(directory)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("static file server did not start")


def launch(p):
    for channel in ("msedge", "chrome", None):
        try:
            return p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except Exception:  # noqa: BLE001 - try the next installed browser
            continue
    raise SystemExit("No Chromium browser found: install Edge/Chrome or run `playwright install chromium`.")


def dashboard_shots(browser) -> None:
    page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=1.5)
    for name, text in DASHBOARD_SHOTS.items():
        page.goto(f"{API}/?q={urllib.parse.quote(text)}")
        page.wait_for_selector("#verdict.bad, #verdict.ok")
        page.wait_for_timeout(500)  # let the score bar animation finish
        page.screenshot(path=OUT / name, full_page=True)
        print("saved", (OUT / name).relative_to(ROOT))
    page.close()


def popup_shots(browser) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(ROOT / "extension", tmp, dirs_exist_ok=True)
        server = serve(Path(tmp), POPUP_PORT)
        try:
            for name, text in POPUP_SHOTS.items():
                page = browser.new_page(viewport={"width": 470, "height": 1320}, device_scale_factor=1.5)
                page.add_init_script(CHROME_STUB % json.dumps(text))
                page.goto(f"http://127.0.0.1:{POPUP_PORT}/index.html")
                page.click("#scanButton")
                page.wait_for_function("document.getElementById('result').innerText.includes('detected')")
                page.click("#checkLinkButton")
                page.wait_for_timeout(500)
                # Crop to the content: the popup's body stretches to the viewport height.
                bottom = page.evaluate("Math.max(...[...document.body.children].map("
                                       "e => e.getBoundingClientRect().bottom))")
                page.screenshot(path=OUT / name, clip={"x": 0, "y": 0, "width": 470, "height": bottom + 24})
                print("saved", (OUT / name).relative_to(ROOT))
                page.close()
        finally:
            server.kill()
            server.wait()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = launch(p)
        try:
            dashboard_shots(browser)
            popup_shots(browser)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
