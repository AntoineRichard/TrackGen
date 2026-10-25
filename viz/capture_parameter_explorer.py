#!/usr/bin/env python3
"""Capture the real explorer's native light/dark UI with one unchanged plot.

Run in a separate environment with the UI extra and the runtime dependencies. Node
and Playwright are external capture tools, not application dependencies::

    PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers python \
        viz/capture_parameter_explorer.py --out /tmp/explorer-capture \
        --node /path/to/node --playwright-module /path/to/playwright

The app runs on loopback only. The browser changes its OS color preference, which
Gradio observes natively; it never injects style rules or regenerates the tracks.
The source must be clean for final published media; --allow-dirty is for previews.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

_BROWSER_CAPTURE = r"""
const fs = require('node:fs');
const crypto = require('node:crypto');
const [moduleName, settingsPath] = process.argv.slice(1);
const { chromium } = require(moduleName);
const options = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({
      viewport: options.viewport, deviceScaleFactor: 1,
      colorScheme: 'light', locale: 'en-US', timezoneId: 'UTC',
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(options.url, { waitUntil: 'domcontentloaded' });
    const plot = page.locator('[data-testid="matplotlib"] img:visible').first();
    await plot.waitFor({ state: 'visible', timeout: 240000 });
    await page.waitForFunction(() => {
      const images = [...document.querySelectorAll('[data-testid="matplotlib"] img')];
      return images.some(img => img.offsetParent && img.complete && img.naturalWidth > 0)
        && document.body.innerText.includes('valid yield:');
    }, null, { timeout: 240000 });
    if (await page.getByText('internal error:', { exact: false }).count()) {
      throw new Error('The real app reports an internal error; capture refused');
    }
    await page.evaluate(() => document.fonts.ready);
    const source = await plot.getAttribute('src');
    if (!source || !source.startsWith('data:image/')) throw new Error('Missing Matplotlib image');
    const payload = Buffer.from(source.slice(source.indexOf(',') + 1), 'base64');
    fs.writeFileSync(options.out + '/plot-payload.' + (source.includes('image/png') ? 'png' : 'webp'), payload);
    const plotHash = sha(payload);
    const records = [];
    for (const theme of ['light', 'dark']) {
      await page.emulateMedia({ colorScheme: theme });
      await page.waitForFunction(wantDark => document.body.classList.contains('dark') === wantDark,
        theme === 'dark', { timeout: 10000 });
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      if (await plot.getAttribute('src') !== source) throw new Error('Plot changed during theme switch');
      await page.screenshot({ path: options.out + '/' + theme + '.png', fullPage: false, animations: 'disabled' });
      records.push(await page.evaluate(theme => {
        const container = document.querySelector('.gradio-container') || document.body;
        return {
          theme, native_dark_class: document.body.classList.contains('dark'),
          background: getComputedStyle(container).backgroundColor,
          body_background: getComputedStyle(document.body).backgroundColor,
          stats: document.body.innerText.match(/valid yield:[^\n]+/)?.[0] || '',
          visible_controls: [...document.querySelectorAll('input,select,textarea')]
            .filter(el => el.offsetParent && el.type !== 'hidden')
            .map(el => ({ label: el.labels?.[0]?.innerText || el.getAttribute('aria-label') || '',
                         type: el.type || el.tagName.toLowerCase(), value: el.value })),
        };
      }, theme));
    }
    if (errors.length) throw new Error('Browser errors: ' + errors.join('; '));
    fs.writeFileSync(options.out + '/browser.json', JSON.stringify({
      browser: browser.version(), node: process.version,
      playwright: require(moduleName + "/package.json").version,
      viewport: options.viewport, device_scale_factor: 1,
      theme_method: 'Gradio system theme via native prefers-color-scheme change',
      screenshot_animations: 'disabled',
      plot_payload_sha256: plotHash, unchanged_plot_payload: true, captures: records,
    }, null, 2) + '\n');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def capture(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    source_commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    if status and not args.allow_dirty:
        raise RuntimeError("Final captures require a clean source tree; use --allow-dirty only for previews")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_files = ["viz/param_explorer.py", "viz/plot_tracks.py", "viz/capture_parameter_explorer.py"]
    source_hashes = {name: _sha(root / name) for name in source_files}
    runtime = hashlib.sha256()
    for path in sorted((root / "track_gen").rglob("*.py")):
        runtime.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    # The capture uses the unchanged application's build_app and load callbacks.
    # Defaults, rendering, source theme, and generation remain application-owned.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    sys.path.insert(0, str(root))
    from viz import param_explorer

    packages = {name: importlib.metadata.version(name) for name in (
        "gradio", "gradio_client", "numpy", "torch", "warp-lang", "matplotlib",
        "fastapi", "pydantic", "huggingface_hub",
    )}
    app = param_explorer.build_app()
    try:
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False,
                   inbrowser=False, prevent_thread_lock=True, show_error=True, quiet=True)
        options = {"url": f"http://127.0.0.1:{args.port}/", "out": str(out),
                   "viewport": {"width": args.width, "height": args.height}}
        with tempfile.TemporaryDirectory(prefix="trackgen-ui-browser-") as tmp:
            settings = Path(tmp) / "settings.json"
            settings.write_text(json.dumps(options))
            subprocess.run([args.node, "-e", _BROWSER_CAPTURE, args.playwright_module, str(settings)],
                           check=True, timeout=300)
    finally:
        app.close()
    if source_commit != _git(root, "rev-parse", "HEAD") or status != _git(root, "status", "--porcelain"):
        raise RuntimeError("Source state changed during capture; rerun from a stable tree")
    if any(_sha(root / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("App or capture source changed during capture")
    browser = json.loads((out / "browser.json").read_text())
    manifest = {
        "source_commit": source_commit, "source_clean": not bool(status),
        "source_status": status.splitlines(), "source_files_sha256": source_hashes,
        "runtime_source_sha256": runtime.hexdigest(), "python_version": sys.version.split()[0],
        "packages": packages, "device": param_explorer.DEVICE,
        "track_controls": param_explorer.default_params(),
        "gate_controls": param_explorer.default_gate_params(),
        "page": 0, "browser": browser,
        "images": {theme: {"path": f"{theme}.png", "sha256": _sha(out / f"{theme}.png")}
                   for theme in ("light", "dark")},
        "presentation": "Native Gradio theme; app-owned Matplotlib plot is unchanged between themes",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--playwright-module", default="playwright")
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1120)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    manifest = capture(args)
    print(json.dumps({"source_commit": manifest["source_commit"], "images": manifest["images"]}, indent=2))


if __name__ == "__main__":
    main()
