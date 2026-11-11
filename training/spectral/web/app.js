"use strict";

const $ = (id) => document.getElementById(id);
const phaseNames = {
  prepare: "Индексация и проверка выборки", train: "Признаки · Train", val: "Признаки · Validation",
  test: "Признаки · Test", ood: "Признаки · OOD", fit: "Обучение классификаторов",
  evaluate: "Расчёт метрик", complete: "Отчёт готов",
};
const splitNames = {train: "Train", val: "Validation", test: "Test", ood: "OOD"};
const modeNames = {spectrum: "Спектр", noise: "Шумовой остаток", combined: "Спектр + шум",
  face: "Только лицо", full_face: "Кадр + лицо"};
const states = {running: "Выполняется", complete: "Завершено", error: "Ошибка",
  stopped: "Остановлено", stale: "Нет обновлений", idle: "Ожидание запуска", prepared: "Выборка готова"};
const percent = (n) => n == null ? "—" : `${(n * 100).toFixed(1).replace(".", ",")} %`;
const value = (n) => n == null ? "—" : String(n);
let report = null;
let reportKey = null;
let spectrum = null;
let image = null;

function rows(target, data, empty, span) {
  const output = data.length ? data : [[empty]];
  target.replaceChildren(...output.map((values) => {
    const row = document.createElement("tr");
    for (const item of values) {
      const cell = document.createElement("td");
      cell.textContent = item;
      if (!data.length) cell.colSpan = span;
      row.append(cell);
    }
    return row;
  }));
}

function drawSpectrum() {
  const canvas = $("spectrum");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const top = 13, bottom = rect.height - 15, left = 43, right = rect.width - 12;
  ctx.font = '11px "Segoe UI", sans-serif';
  ctx.fillStyle = "#52645e";
  ctx.strokeStyle = "#d7e2de";
  for (const db of [0, -20, -40, -60, -80]) {
    const y = top + (bottom - top) * (-db / 80);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.fillText(String(db), 8, y + 3);
  }
  if (!spectrum?.length) return;
  const gap = (right - left) / spectrum.length;
  spectrum.forEach((logEnergy, i) => {
    const height = Math.max(1, (bottom - top) * (1 + Math.max(-8, Math.min(0, logEnergy)) / 8));
    ctx.fillStyle = i < 5 ? "#277c69" : "#367695";
    ctx.fillRect(left + gap * i + 2, bottom - height, Math.max(1, gap - 4), height);
  });
}

function renderReport() {
  const split = $("split").value;
  const variants = report?.variants;
  const data = Object.entries(modeNames).flatMap(([mode, label]) => {
    const clean = variants?.[mode]?.[`${split}/clean`];
    const jpeg = variants?.[mode]?.[`${split}/jpeg70`];
    if (!clean) return [];
    const pairs = clean.dfdc_pairs;
    return [[label, percent(clean.balanced_accuracy), percent(jpeg?.balanced_accuracy),
      percent(clean.recall_fake), percent(clean.false_positive_rate),
      pairs?.complete ? `${pairs.fake_ranked_higher}/${pairs.complete}` : "—", `${clean.n}/${clean.requested}`]];
  });
  rows($("metrics"), data, variants ? "Нет результатов для этого раздела" : "Отчёт ещё не готов", 7);
  $("download").hidden = !variants;
}

function renderPreview(preview, historicalSpectrum) {
  const src = preview?.image ? `data:image/jpeg;base64,${preview.image}` : null;
  if (src !== image) {
    if (src) $("sample-frame").src = src;
    else $("sample-frame").removeAttribute("src");
    image = src;
  }
  $("sample-frame").hidden = !src;
  $("frame-empty").hidden = !!src;
  $("sample-split").textContent = splitNames[preview?.split] || "";
  $("sample-time").textContent = preview ? `${preview.timestamp.toFixed(2)} с · ${preview.width}×${preview.height}` : "";
  $("sample-name").textContent = preview ? `${preview.source} · ${preview.name}` : "";
  $("sample-prediction").textContent = preview?.prediction || "—";
  $("sample-prediction").dataset.label = preview?.prediction || "";
  $("sample-label").textContent = preview?.label || "—";
  $("sample-score").textContent = preview?.score == null ? "—" : preview.score.toFixed(4);
  $("sample-threshold").textContent = preview?.threshold == null ? "—" : preview.threshold.toFixed(4);
  $("sample-note").textContent = preview?.prediction ? "Score рассчитан по всем выбранным кадрам ролика." :
    "Оценка появится после обучения и подбора порога.";
  $("face-note").hidden = preview?.face_frames == null;
  const faces = preview?.face_frames;
  const found = preview?.face_box ? "Рамка отмечает область анализа, не подмену." : "На этом кадре лицо не выбрано.";
  const scores = preview?.face_score != null ?
    `Score лица: ${preview.face_score.toFixed(4)} · Score кадра: ${preview.full_score.toFixed(4)}.` :
    preview?.model === "full_face" ? "Решение только по кадру." : "";
  $("face-note").textContent = faces == null ? "" : `Лицо: ${faces}/${preview.frames} кадров. ${found} ${scores}`;
  spectrum = preview?.spectrum || historicalSpectrum;
  $("spectrum-state").textContent = preview ? "Доля энергии по полосам, дБ" :
    spectrum?.length ? "Спектр сохранённого запуска без кадра" : "Нет извлечённых признаков";
  drawSpectrum();
}

function render(state) {
  const stale = state.status === "running" && Date.now() / 1000 - state.updated > 20;
  const status = stale ? "stale" : state.status || (report?.variants ? "complete" : "idle");
  $("state").dataset.state = status;
  $("state").textContent = states[status] || states.idle;
  $("run-name").textContent = state.run ? state.run.split(/[/\\]/).at(-1) : "spectral-full-frame-v2";
  $("stage").textContent = state.detail === "dfdc_hashes" ? "Проверка файлов DFDC" :
    phaseNames[state.phase] || (status === "complete" ? phaseNames.complete : "Подготовка выборки");
  const details = state.phase === "fit" ? modeNames[state.active_variant] : state.phase === "evaluate" ? state.evaluation : "";
  $("stage-count").textContent = state.total ? `${state.completed} / ${state.total}${details ? ` · ${details}` : ""}` : "—";
  $("progress").value = status === "complete" ? 1 : state.total ? state.completed / state.total : 0;
  const steps = [...$("steps").children];
  const active = steps.findIndex((li) => li.dataset.phase === state.phase);
  steps.forEach((li, i) => {
    li.dataset.stage = status === "complete" || i < active ? "done" : i === active ? "current" : "";
  });
  for (const id of ["usable", "cached", "failed"]) $(id).textContent = value(state[id]);
  $("usable-label").textContent = status === "complete" ? "Всего обработано" : "Обработано на этапе";
  const end = state.status === "running" ? state.updated : state.finished;
  const seconds = state.started && end ? Math.max(0, Math.floor(end - state.started)) : null;
  $("elapsed").textContent = seconds == null ? "—" : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  $("cpu").textContent = percent(state.system?.cpu_percent == null ? null : state.system.cpu_percent / 100);
  $("ram").textContent = percent(state.system?.ram_percent == null ? null : state.system.ram_percent / 100);
  $("face-coverage").hidden = state.face_videos == null;
  $("face-videos").textContent = value(state.face_videos);
  rows($("cohort"), Object.entries(state.plan || {}).map(([split, count]) =>
    [splitNames[split], String(count), value(state.dfdc?.[split])]), "Выборка ещё не подготовлена", 3);
  renderPreview(state.preview, state.spectrum);
}

async function get(route, type = "json") {
  const response = await fetch(route, {cache: "no-store"});
  if (!response.ok) throw new Error(String(response.status));
  return response[type]();
}

async function refresh() {
  try {
    const state = await get("/api/state");
    const key = state.status === "complete" ? state.finished : !state.status ? "historic" : null;
    if (key === null) {
      report = null;
      reportKey = null;
    } else if (key !== reportKey) {
      report = await get("/api/report");
      reportKey = report.variants ? key : null;
    }
    render(state);
    renderReport();
    $("connection").textContent = "Локальный монитор";
    const log = await get("/api/log", "text");
    if ($("logs").textContent !== log && log) {
      const position = $("logs").scrollTop;
      $("logs").textContent = log;
      $("logs").scrollTop = $("follow").checked ? $("logs").scrollHeight : position;
    }
  } catch {
    $("connection").textContent = "Ожидание связи";
  } finally {
    window.setTimeout(refresh, 1500);
  }
}

$("split").addEventListener("change", renderReport);
new ResizeObserver(drawSpectrum).observe($("spectrum"));
window.lucide?.createIcons();
refresh();
