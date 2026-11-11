const assert = require("node:assert/strict");
const { chromium } = require("playwright");

(async () => {
  const url = process.env.TRACE_SPECTRAL_URL || "http://127.0.0.1:8767";
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(url);
    if (process.env.TRACE_SPECTRAL_LIVE) {
      await page.waitForSelector('#state[data-state="idle"]');
      console.log("Waiting view verified; launch the matching training command now.");
      await page.waitForSelector('#state[data-state="running"]', { timeout: 90000 });
      await page.waitForFunction(() => Number(document.querySelector("#progress").value) > 0);
      console.log("Live progress verified.");
    }
    await page.waitForSelector('#state[data-state="complete"]', { timeout: 120000 });
    const variants = Number(process.env.TRACE_SPECTRAL_VARIANTS || 3);
    await page.waitForFunction(n => document.querySelectorAll("#metrics tr").length === n, variants);
    await page.waitForFunction(() => document.querySelector("#logs").textContent.includes("COMPLETE"));
    assert.equal(await page.locator("#progress").evaluate(node => node.value), 1);
    assert.match(await page.locator("#logs").innerText(), /COMPLETE/);
    assert.equal(await page.locator("#download").isVisible(), true);
    await page.waitForFunction(() => document.querySelector("#sample-frame").naturalWidth > 0);
    assert.match(await page.locator("#sample-prediction").innerText(), /^(REAL|FAKE)$/);
    assert.match(await page.locator("#sample-label").innerText(), /^(REAL|FAKE)$/);
    const score = Number(await page.locator("#sample-score").innerText());
    const threshold = Number(await page.locator("#sample-threshold").innerText());
    if (Math.abs(score - threshold) > .0001) {
      assert.equal(await page.locator("#sample-prediction").innerText(), score >= threshold ? "FAKE" : "REAL");
    }
    if (variants === 5) {
      assert.match(await page.locator("#face-note").innerText(), /Score лица:/);
      assert.match(await page.locator("#metrics").innerText(), /Кадр \+ лицо/);
    }
    const hasBars = await page.locator("canvas").evaluate(canvas => {
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      return data.some((v, i) => i % 4 === 0 && v === 39 && data[i+1] === 124 && data[i+2] === 105);
    });
    assert.ok(hasBars, "Expected real spectrum bars on the canvas");
    await page.selectOption("#split", "val");
    assert.equal(await page.locator("#metrics tr").count(), variants);
    if (process.env.TRACE_SPECTRAL_SCREENSHOT) {
      await page.screenshot({ path: process.env.TRACE_SPECTRAL_SCREENSHOT, fullPage: true });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth === innerWidth));
    assert.deepEqual(errors, []);
    console.log("Spectral dashboard verified on desktop and mobile.");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
