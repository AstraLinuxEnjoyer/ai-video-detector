"""Self-contained, printable local analysis report."""

from base64 import b64encode
from html import escape


def number(value):
    return '—' if value is None else f'{value:.3f}'.replace('.', ',')


def picture(directory, name, title):
    if not name:
        return ''
    path = directory / name
    if path.parent.resolve() != directory.resolve() or not path.is_file() or path.suffix not in {'.jpg', '.gif'}:
        return ''
    mime = 'jpeg' if path.suffix == '.jpg' else 'gif'
    data = b64encode(path.read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/{mime};base64,{data}" alt="{escape(title)}"><figcaption>{escape(title)}</figcaption></figure>'


def chart(frames, clips, duration, threshold, regions):
    width, height = 760, 150
    x = lambda t: 32 + min(1, max(0, t/max(duration, .001))) * (width-44)
    y = lambda score: 10 + (1-score) * (height-33)
    parts = [f'<rect x="{x(r["start"]):.2f}" y="10" width="{x(r["end"])-x(r["start"]):.2f}" height="117" fill="#f8dcd1"/>' for r in regions]
    for tick in (0, .5, 1):
        parts.append(f'<text x="3" y="{y(tick)+4:.2f}">{tick:g}</text><path d="M32 {y(tick):.2f}H748" stroke="#dbe1df"/>')
    parts.append(f'<path d="M32 {y(threshold):.2f}H748" stroke="#a65043" stroke-dasharray="4 4"/>')
    for values, color in (([(f['time'], f['spatial']) for f in frames], '#387452'),
                          ([((c['start']+c['end'])/2, c['temporal']) for c in clips], '#4265a3')):
        coords = [(x(t), y(s)) for t, s in values]
        points = ' '.join(f'{a:.2f},{b:.2f}' for a, b in coords)
        parts.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')
        parts.extend(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="2" fill="{color}"/>' for a, b in coords)
    parts.append(f'<text x="32" y="145">0 с</text><text x="748" y="145" text-anchor="end">{duration:.1f} с</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Оценки по времени">{"".join(parts)}</svg>'


def spectrum_chart(values):
    bars = []
    for i, value in enumerate(values):
        h = max(0., min(80., 80.+value))
        bars.append(f'<rect x="{i*31+25}" y="{88-h:.2f}" width="25" height="{h:.2f}" fill="#81639c"/>')
    return ('<svg viewBox="0 0 345 110" role="img" aria-label="Спектр кадра">'
            '<text x="1" y="12">0</text><text x="0" y="88">−80</text>' + ''.join(bars) +
            '<text x="25" y="105">Низкие частоты</text><text x="335" y="105" text-anchor="end">Высокие частоты</text></svg>')


def render_report(job, directory):
    summary, meta = job['summary'], job['metadata']
    spectral = job.get('spectral') or {}
    frames, clips = job.get('frames', []), job.get('clips', [])
    regions = summary.get('segments', [])
    reviews = summary.get('review_segments', [])
    verdicts = {'suspected': 'Возможное вмешательство', 'uncertain': 'Нужна дополнительная проверка',
                'no_signals': 'Сильных признаков не найдено'}
    reasons = {'cross_branch_intervals': 'Кадр и движение дали высокий сигнал в одном участке.',
               'single_branch_persistence': 'Повторяется сигнал одной модели.',
               'isolated_high_scores': 'Отдельные высокие отсчёты без подтверждения.',
               'elevated_mean_score': 'Повышенный средний балл.', 'branch_disagreement': 'Оценки веток расходятся.',
               'motion_not_complete': 'Движение оценено не полностью.', 'spectral_review': 'Спектр дал повод для проверки.',
               'low_scores_with_complete_motion': 'Нет оснований для тревоги в проверенных отсчётах.'}
    reason = ' '.join(reasons.get(key, '') for key in summary.get('reasons', []))
    weights = summary.get('weights', {})
    labels = {'spatial': 'Изображение', 'temporal': 'Движение', 'spectral': 'Спектр'}
    rows = ''.join(f'<tr><th>{label}</th><td>{number(summary.get(key))}</td><td>{weights.get(key, 0):.0%}</td></tr>'
                   for key, label in labels.items())
    listing = ''.join(f'<li>{r["start"]:.2f}–{r["end"]:.2f} с · {label}</li>' for spans, label in
                      ((regions, 'совпадение двух веток'), (reviews, 'ручная проверка')) for r in spans)
    supported = [f for f in frames if any(s['start'] <= f['time'] < s['end'] for s in regions+reviews)]
    chosen = sorted(supported or [f for f in frames if f.get('heatmap')], key=lambda f: f['spatial'], reverse=True)[:4]
    images = ''.join(picture(directory, f['image'], f"Кадр {f['time']:.2f} с · балл {number(f['spatial'])}") +
                     picture(directory, f.get('heatmap') if f.get('heat_valid') else None,
                             f"Grad-CAM · {f['time']:.2f} с") for f in chosen)
    top_clip = max(clips, key=lambda c: c['temporal'], default=None)
    motion = picture(directory, top_clip.get('flow'), f"{top_clip['start']:.2f}–{top_clip['end']:.2f} с · {number(top_clip['temporal'])}") if top_clip else ''
    spectra = ''.join('<div>' + picture(directory, s['image'], f"Яркость · {s['time']:.2f} с") +
                      spectrum_chart(s['bands_db']) + '</div>' for s in spectral.get('samples', []))
    variants = spectral.get('variants', {})
    variants_table = ''.join(f'<tr><th>{label}</th><td>{number(variants[key]["score"])}</td><td>{number(variants[key]["threshold"])}</td></tr>'
                             for key, label in (('spectrum', 'Только спектр'), ('noise', 'Шумовой остаток'),
                                 ('combined', 'Полный кадр'), ('face', 'Только лицо'), ('full_face', 'Кадр + лицо'))
                             if key in variants)
    variants_table = '<table><tr><th>Вариант</th><th>Балл</th><th>Порог</th></tr>'+variants_table+'</table>' if variants_table else ''
    warnings = ''.join(f'<li>{escape(warning)}</li>' for warning in job.get('warnings', []))
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRACE · Отчёт анализа</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;color:#202925;margin:0;background:#f3f5f3}}main{{max-width:940px;margin:auto;background:white;padding:clamp(18px,5%,48px)}}header{{border-top:4px solid #427355;padding-top:18px}}h1,h2{{line-height:1.2}}h2{{border-top:1px solid #dde3df;padding-top:16px;margin-top:30px}}p,td{{overflow-wrap:anywhere}}small,figcaption{{color:#53625a}}table{{border-collapse:collapse;width:100%;text-align:left}}th,td{{border-bottom:1px solid #dde3df;padding:7px}}svg{{width:100%;height:auto;font-size:11px}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:14px}}figure{{margin:0;break-inside:avoid}}img{{width:100%;max-height:340px;object-fit:contain;background:#18221d;border-radius:5px}}@media print{{body{{background:white}}main{{padding:0}}h2{{break-after:avoid}}}}
</style><main><header><small>TRACE / ЛОКАЛЬНЫЙ АНАЛИЗ</small><h1>{verdicts.get(summary['verdict'], 'Результат проверки')}</h1>
<p><strong>{escape(job['filename'])}</strong> · {meta['width']} × {meta['height']} · {meta['duration']:.2f} с</p><p>{reason}</p>
<p>Общий балл: <strong>{number(summary['combined'])}</strong> · правило {escape(summary.get('policy', 'TRACE'))}</p></header>
<h2>Вклад моделей</h2><table><tr><th>Модель</th><th>Балл</th><th>Вес</th></tr>{rows}</table>
<p>Кадры: {len(frames)} · клипы: {len(clips)}/{summary.get('coverage', {}).get('expected_clips', len(clips))} · лицо в {spectral.get('face_frames', 0)} спектральных кадрах.</p>
<h2>График по времени</h2><p><small>Зелёный — изображение; синий — движение; пунктир — фиксированный порог {number(summary['threshold'])}. У спектра один балл на видео, не линия во времени.</small></p>
{chart(frames, clips, meta['duration'], summary['threshold'], regions)}<ul>{listing or '<li>Выделенных интервалов нет.</li>'}</ul>
<h2>Изображение и карты влияния</h2><p>Примеры для проверки. Grad-CAM показывает области влияния на оценку, а не маску подмены.</p><div class="images">{images}</div>
<h2>Движение</h2><p>Оптический поток клипа с наибольшим баллом. Цвет — направление, яркость — величина движения. Это вход модели, не карта внимания.</p><div class="images">{motion or 'Клипы движения недоступны.'}</div>
<h2>Спектр</h2><p>Кадры яркости, энергия частотных полос (дБ) и область лица, если она найдена. Рамка не обозначает подмену.</p><div class="images">{spectra or 'Спектральный анализ недоступен.'}</div>{variants_table}
{f'<h2>Ограничения этой проверки</h2><ul>{warnings}</ul>' if warnings else ''}
<p><small>Оценки не являются вероятностями. Результат не доказывает ни вмешательство, ни подлинность видео.</small></p></main></html>'''
