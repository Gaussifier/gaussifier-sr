#!/usr/bin/env node
// Drive the built super-resolution demo end to end, as served from a sub-path: load the page,
// wait for the automatic x2 run of a Kodak sample, switch to x4, flip the comparison, upload an
// LR image, and read the stats after every run. Every request is checked to stay under the
// sub-path (the fonts excepted) and to succeed.
//
//   node tools/demosr-check.mjs                                  # serves apps/demosr/dist at /demosr/
//   node tools/demosr-check.mjs --dir ../site --base /demosr/ [--sample kodim03] [--upload lr.png] [--shot prefix] [--theme light] [--viewport 390x844]
//   node tools/demosr-check.mjs --url https://host:8443/demosr/
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium } from "./chromium.mjs";
import { out, parseArgs } from "./cli.mjs";
import { createHandler } from "./server.mjs";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = parseArgs(process.argv.slice(2), { url: null, dir: path.join(web, "apps/demosr/dist"), base: "/demosr/", sample: "kodim03", upload: null, shot: null, theme: null, viewport: "1440x900" });

/** Serve `dir` at `base` only, the way GitHub Pages serves a project site. */
function serveAt(dir, base) {
  const handler = createHandler(dir);
  const server = http.createServer((req, res) => {
    const url = req.url.split("?")[0];
    if (url === base.slice(0, -1)) { res.writeHead(301, { Location: base + req.url.slice(url.length) }); res.end(); return; }
    if (!url.startsWith(base)) { res.writeHead(404); res.end("outside the site"); return; }
    req.url = req.url.slice(base.length - 1);
    handler(req, res);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port })));
}

const runsDone = (page) => page.evaluate(() => Number(document.getElementById("runStatus").dataset.runs ?? 0));
const waitForRun = (page, n) => page.waitForFunction(
  (n) => Number(document.getElementById("runStatus").dataset.runs ?? 0) >= n && /^(Done|Run failed)/.test(document.getElementById("runStatus").textContent),
  { timeout: 300000 }, n);
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

async function readStats(page) {
  return page.evaluate(() => {
    const t = (id) => document.getElementById(id).textContent;
    return {
      image: t("imageName"), card: t("imageDims"), status: t("runStatus"), psnr: t("psnr"), bicubic: t("psnrBicubic"),
      gaussians: t("statN"), passes: document.getElementById("statN").title, time: t("time"),
      scale: document.querySelector('#scale button[aria-pressed="true"]')?.textContent,
      left: t("labelLeft"), right: t("labelRight"), pill: t("gpuPill"),
    };
  });
}

/** Click a button and wait for the run it starts. */
async function clickAndRun(page, selector) {
  const before = await runsDone(page);
  await page.click(selector);
  await waitForRun(page, before + 1);
}

async function main() {
  const { browser, close } = await launchChromium({ ignoreCertErrors: !!args.url, protocolTimeout: 600000 });
  const local = args.url ? null : await serveAt(path.resolve(args.dir), args.base);
  const root = args.url ?? `http://127.0.0.1:${local.port}${args.base}`;
  const requests = [], failures = [];
  const steps = {};
  let ok = true;
  try {
    const page = await browser.newPage();
    const [vw, vh] = args.viewport.split("x").map(Number);
    await page.setViewport({ width: vw, height: vh, isMobile: vw < 600, hasTouch: vw < 600 });
    if (args.theme) await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: args.theme }]);
    page.setDefaultTimeout(600000);
    page.on("console", (m) => { const t = m.text(); if (!/favicon|onnxruntime:/.test(t)) out("  [page] " + t.slice(0, 300)); });
    page.on("pageerror", (e) => { ok = false; out("  [pageerror] " + String(e).slice(0, 300)); });
    page.on("requestfinished", (r) => { const s = r.response()?.status() ?? 0; requests.push({ url: r.url(), status: s }); if (s >= 400) failures.push(`${s} ${r.url()}`); });
    page.on("requestfailed", (r) => failures.push(`failed ${r.url()} ${r.failure()?.errorText ?? ""}`));
    // The trailing slash is what Pages redirects to; load the bare path to exercise that too.
    await page.goto((args.url ? root : root.slice(0, -1)) + `?sample=${args.sample}`);
    await waitForRun(page, 1);
    steps.x2 = await readStats(page);
    if (args.shot) { await pause(400); await page.screenshot({ path: `${args.shot}-x2.png` }); }

    await clickAndRun(page, '#scale button[data-scale="4"]');
    steps.x4 = await readStats(page);
    await page.click('#compare button[data-compare="original"]');
    await pause(100);
    steps.original = (await readStats(page)).left;
    await page.click('#compare button[data-compare="input"]');
    await pause(100);
    steps.input = (await readStats(page)).left;
    await page.click('#compare button[data-compare="bicubic"]');
    // Drag the divider to a quarter of the stage, as a user would.
    const box = await (await page.$("#view")).boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2, { steps: 5 });
    await page.mouse.up();
    if (args.shot) { await pause(400); await page.screenshot({ path: `${args.shot}-x4.png` }); }
    // Zoom in on the input next to the result, 4x around a point left of the divider.
    await page.click('#compare button[data-compare="input"]');
    await page.mouse.move(box.x + box.width * 0.32, box.y + box.height * 0.45);
    for (let i = 0; i < 4; i++) await page.mouse.wheel({ deltaY: -230 });
    if (args.shot) { await pause(400); await page.screenshot({ path: `${args.shot}-zoom.png` }); }
    await page.keyboard.press("f");
    await page.click('#compare button[data-compare="bicubic"]');

    if (args.upload) {
      const before = await runsDone(page);
      await (await page.$("#image")).uploadFile(path.resolve(args.upload));
      await waitForRun(page, before + 1);
      steps.upload = await readStats(page);
      steps.upload.original_disabled = await page.$eval('#compare button[data-compare="original"]', (b) => b.disabled);
    }
    // The fonts come from Google Fonts as in the regular demo; the browser asks for /favicon.ico on its own.
    const outside = requests.filter((r) => !r.url.startsWith(root.replace(/\/$/, "")) && !/^blob:|fonts\.(googleapis|gstatic)\.com|\/favicon\.ico$/.test(r.url));
    for (let i = failures.length - 1; i >= 0; i--) if (/\/favicon\.ico$/.test(failures[i])) failures.splice(i, 1);
    const done = (s) => s && s.status.startsWith("Done");
    ok = ok && done(steps.x2) && done(steps.x4) && (!args.upload || done(steps.upload)) && failures.length === 0 && outside.length === 0;
    const loaded = [...new Set(requests.map((r) => r.url.replace(root, "").split("?")[0]))].filter((u) => !/^(samples|assets)\//.test(u) || /index/.test(u));
    out(JSON.stringify({ ok, root, steps, failures, outside: outside.map((r) => r.url), loaded }, null, 1));
  } catch (e) {
    ok = false;
    out(JSON.stringify({ ok, root, steps, error: String(e).slice(0, 800), failures }, null, 1));
  } finally {
    await close();
    local?.server.close();
  }
  process.exit(ok ? 0 : 1);
}

main().catch((e) => { out(String(e)); process.exit(1); });
