// Integration check against the local read-only dashboard with real smoke artifacts.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
(async () => {
  const out = path.resolve('cache/temporal_v2/ui-test');
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport);
      await page.goto(process.env.DASHBOARD_URL || 'http://127.0.0.1:8767');
      await page.waitForFunction(() => document.querySelector('#input').naturalWidth > 0);
      const state = await (await page.request.get((process.env.DASHBOARD_URL || 'http://127.0.0.1:8767') + '/api/state')).json();
      if (process.env.REQUIRE_MOTION === '1') {
        assert.equal(state.config.input_mode, 'flow');
        assert((await page.locator('#input-caption').innerText()).includes('Оптический поток'));
        const nonblank = await page.locator('#input').evaluate(image => {
          const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
          const c=canvas.getContext('2d');c.drawImage(image,0,0);
          return c.getImageData(0,0,canvas.width,canvas.height).data.some((v,i)=>i%4!==3&&v>0);
        });
        assert(nonblank, 'missing flow colors');
      }
      for (const tab of ['live', 'quality', 'explain', 'system']) {
        await page.locator(`[data-tab="${tab}"]`).click();
        await page.waitForFunction(() => [...document.querySelectorAll('.tab.active img')].every(i => i.complete && i.naturalWidth > 0));
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'page overflow');
        if (tab === 'live' || tab === 'quality') {
          const nonblank = await page.locator(tab === 'live' ? '#loss-chart' : '#quality-chart').evaluate(canvas => {
            const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
            return data.some((v, i) => i % 4 === 3 && v > 0);
          });
          assert(nonblank, 'blank chart');
        }
        if (tab === 'explain') {
          await page.locator('#explain-select').selectOption('0');
          await page.waitForFunction(() => document.querySelector('#explain-epoch').textContent === 'Эпоха 0');
          await page.locator('#explain-select').selectOption('latest');
          await page.waitForFunction(epoch => document.querySelector('#explain-epoch').textContent === 'Эпоха ' + epoch, state.explanation.epoch);
        }
        if (tab === 'quality') {
          if (process.env.REQUIRE_MOTION === '1') {
            assert(await page.locator('#motion-controls').isVisible());
            assert.equal(await page.locator('#motion-control-rows tr').count(), 3);
          }
          for (const split of ['test', 'ood']) {
            const coverage = state.final?.[split]?.coverage;
            if (!coverage) continue;
            const summary = await page.locator('#final').innerText();
            assert(summary.includes(`Проверено ${coverage.evaluated} из ${coverage.requested}`), 'missing requested denominator');
            assert(summary.includes(`без ответа ${coverage.unsupported}`), 'missing abstentions');
          }
        }
        await page.screenshot({ path: path.join(out, `${viewport.width}-${tab}.png`), fullPage: true });
      }
    }
    assert.deepEqual(errors, []);
    console.log('TEMPORAL_DASHBOARD_OK desktop/mobile, four tabs, images, canvas, no JS errors');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
