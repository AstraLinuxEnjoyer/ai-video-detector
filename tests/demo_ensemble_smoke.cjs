/* Run against an already started TRACE server with the local trained weights. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const {chromium} = require('playwright');

const base = process.env.TRACE_DEMO_URL || 'http://127.0.0.1:8787';
const output = process.env.TRACE_QA_DIR || 'reports';

async function analyze(page, file) {
  await page.locator('#file-input').setInputFiles(file);
  await page.waitForFunction(() => document.body.dataset.state === 'complete' ||
    document.body.dataset.state === 'error', null, {timeout: 120000});
  const id = await page.evaluate(() => localStorage.getItem('trace-last-job'));
  const url = `${base}/api/jobs/${id}`;
  const job = await (await page.request.get(url)).json();
  assert.equal(job.status, 'complete', job.message);
  assert.equal(job.summary.policy, 'trace-ensemble-v1');
  assert.equal(job.spectral.status, 'complete', JSON.stringify(job.warnings));
  assert.equal(job.spectral.samples.length, 3);
  assert.equal(job.summary.weights.spectral, .1);
  assert.ok(Math.abs(job.summary.combined-(.9*job.summary.base_combined+.1*job.spectral.score))<1e-8);
  return {job, url};
}

(async () => {
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:900}, reducedMotion:'reduce'});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(base);
    const {job, url} = await analyze(page, 'test_videos/real/3.mp4');
    assert.equal(await page.locator('input[type=range]').count(), 0);
    await page.locator('#explore-button').click();
    await page.locator('#detail-image').evaluate(img => img.decode());
    await page.locator('#evidence-image').evaluate(img => img.decode());
    assert.ok((await page.locator('#evidence-image').getAttribute('src')).includes('heat_'));
    await page.locator('[data-view=temporal]').click();
    await page.waitForFunction(() => document.querySelector('#evidence-image').src.includes('flow_'));
    await page.locator('#evidence-image').evaluate(img => img.decode());
    await page.locator('[data-view=spectral]').click();
    await page.waitForFunction(() => document.querySelector('#detail-image').src.includes('spectral_'));
    await page.locator('#detail-image').evaluate(img => img.decode());
    assert.equal(await page.locator('#spectrum-chart rect').count(), 10);
    assert.equal(await page.locator('#sample-picker option').count(), 3);
    await page.locator('#sample-picker').selectOption('2');
    await page.waitForFunction(() => document.querySelector('#detail-image').src.includes('spectral_02'));
    for(const [name,w,h] of [['desktop',1440,900],['mobile',390,844]]) {
      await page.setViewportSize({width:w,height:h});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name} overflow`);
      await page.locator('.report-inspector').screenshot({path:`${output}/ensemble-${name}.png`});
    }
    await page.locator('#report-button').click();
    const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#html-download').click()]);
    assert.equal(download.suggestedFilename(), 'report.html');
    const text = await fs.readFile(await download.path(), 'utf8');
    assert.match(text, /data:image\/jpeg;base64,/);
    assert.match(text, /Энергия|энергия/);
    assert.match(text, /Оптический поток/);
    assert.match(text, /<svg/);
    const offline = await browser.newPage();
    await offline.context().setOffline(true);
    await offline.setContent(text);
    assert.ok(await offline.locator('img').first().evaluate(async img => { await img.decode(); return img.naturalWidth > 0; }));
    assert.equal((await page.request.get(`${url}/files/input.video`)).status(), 404);
    await page.goto(base);
    const second = await analyze(page, 'test_videos/fake/336852_small.mp4');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify([job,second.job].map(r=>({id:r.id,file:r.filename,verdict:r.summary.verdict,
      spectral:r.spectral.score,faceFrames:r.spectral.face_frames,combined:r.summary.combined}))));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
