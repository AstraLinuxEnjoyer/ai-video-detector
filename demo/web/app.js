/* Local-only interface; every plotted value comes from the inference service. */
const $ = id => document.getElementById(id);
let job = null, selected = 0, view = 'frame', threshold = .65, timer = null, started = 0;
let uploadRequest = null, uploading = false, lastFrameCount = -1;
const phases = ['decode','spatial','temporal','explain'];
const names = {decode:'Извлекаем кадры',spatial:'Анализ структуры',temporal:'Анализ движения',explain:'Карты влияния',models:'Загрузка весов',complete:'Анализ завершен'};
const evidenceLabels = {cross_branch:'Обе модели',spatial_persistence:'Соседние кадры',motion_persistence:'Соседние клипы'};
const reviewReasons = {isolated_high_scores:'Есть высокий балл без подтверждения.',single_branch_persistence:'Сигнал повторяется только в одной ветке.',elevated_mean_score:'Средний балл не низкий.',branch_disagreement:'Оценки кадра и движения расходятся.',motion_not_complete:'Проверка движения неполная или недоступна.'};
const icons = () => window.lucide?.createIcons();
const time = seconds => {const s = Math.max(0, seconds || 0); return `${Math.floor(s/60).toString().padStart(2,'0')}:${Math.floor(s%60).toString().padStart(2,'0')}`;};
const score = value => Number.isFinite(value) ? value.toFixed(3) : '—';
const fileURL = name => `/api/jobs/${job.id}/files/${encodeURIComponent(name)}`;
const busy = () => uploading || job && ['running','queued'].includes(job.status);
function notice(message){ $('notice').textContent = message; $('notice').hidden = !message; }
function progress(value){ $('progress-fill').style.width = `${Math.min(100,Math.max(0,value))}%`; }
async function responseJSON(response){const data = await response.json(); if(!response.ok) throw new Error(data.detail || 'Ошибка сервера'); return data;}

async function health(){
  try{const data = await responseJSON(await fetch('/api/health')); const ready = data.status==='ready'&&data.ffmpeg;
    $('server-status').textContent = ready ? `Локально · ${data.device.toUpperCase()}` : 'Требуется настройка';
    $('server-dot').className = `status-dot ${ready?'ready':'error'}`;
    if(!ready) notice(data.message || 'FFmpeg не найден. Запустите установку демонстрации.');
    const info = $('model-info'); info.replaceChildren();
    for(const [kind,title] of [['spatial','EfficientNet V2-S / hybrid'],['temporal','R(2+1)D-18 / motion flow']]){
      const row = document.createElement('div'); row.className = 'model-info-row';
      const strong=document.createElement('strong'); strong.textContent=title;
      const description=document.createElement('span'); description.textContent=data.models?.[kind] ? `Лучшие веса · эпоха ${data.models[kind].epoch}` : 'Веса отсутствуют';
      const hash=document.createElement('code'); hash.textContent=data.models?.[kind]?.sha256 || '';
      row.append(strong,description,hash); info.append(row);
    }
  }catch(error){$('server-status').textContent='Нет подключения';$('server-dot').className='status-dot error';notice(error.message);}
}

function reset(){
  clearTimeout(timer); job=null;selected=0;view='frame';lastFrameCount=-1;
  $('player').pause(); $('player').removeAttribute('src');$('player').load();$('player').hidden=true;
  $('empty-state').hidden=false;$('scan-overlay').hidden=true;progress(0);
  $('dropzone').setAttribute('role','button');$('dropzone').setAttribute('tabindex','0');
  $('report-button').disabled=true;$('filmstrip').replaceChildren();$('file-name').textContent='Новый анализ';
  $('video-meta').textContent='—';$('duration').textContent='—'; render();
}

function upload(file){
  if(!file) return;
  if(busy()){notice('Сначала завершите или остановите текущий анализ.');return;}
  if(file.size>400*1024*1024){notice('Максимальный размер: 400 МБ.');return;}
  reset();notice('');uploading=true;started=Date.now();$('file-name').textContent=file.name;
  $('process-status').textContent='Загрузка файла';$('cancel-button').hidden=false;
  $('new-button').disabled=true;$('choose-button').disabled=true;
  const form=new FormData();form.append('file',file);const request=new XMLHttpRequest();uploadRequest=request;
  request.open('POST','/api/jobs');request.responseType='json';
  request.upload.onprogress=event=>{if(event.lengthComputable){progress(event.loaded/event.total*100);$('process-status').textContent=`Загрузка файла · ${Math.round(event.loaded/event.total*100)}%`;}};
  request.onload=()=>{uploading=false;uploadRequest=null;$('new-button').disabled=false;$('choose-button').disabled=false;
    if(request.status!==202){notice(request.response?.detail || 'Не удалось загрузить видео');$('cancel-button').hidden=true;return;}
    job={id:request.response.id,status:'queued',phase:'models',frames:[],clips:[]};
    localStorage.setItem('trace-last-job',job.id);progress(0);poll();};
  request.onerror=()=>finishUploadError('Соединение с сервером прервано');
  request.onabort=()=>finishUploadError('Загрузка отменена');request.send(form);
}
function finishUploadError(message){uploading=false;uploadRequest=null;$('new-button').disabled=false;$('choose-button').disabled=false;$('cancel-button').hidden=true;notice(message);$('process-status').textContent=message;}

async function poll(){
  if(!job) return;
  try{job=await responseJSON(await fetch(`/api/jobs/${job.id}`));render();
    if(['queued','running'].includes(job.status))timer=setTimeout(poll,800);
  }catch(error){notice(`Связь с сервером: ${error.message}. Результат можно восстановить через «Последний анализ».`);$('scan-overlay').hidden=true;$('cancel-button').hidden=true;}
}

function render(){
  const running=job&&['running','queued'].includes(job.status);
  $('cancel-button').hidden=!running;$('new-button').disabled=!!running;$('choose-button').disabled=!!running;
  $('scan-overlay').hidden=!running;$('scan-text').textContent=names[job?.phase]||'Подготовка';
  $('report-button').disabled=job?.status!=='complete';
  if(job){$('file-name').textContent=job.filename||$('file-name').textContent;progress(job.progress||0);$('process-status').textContent=job.message||names[job.phase]||job.status;
    $('elapsed').textContent=time(job.summary?.elapsed_seconds || (Date.now()-started)/1000);
    if(job.status==='error'||job.status==='cancelled')notice(job.message);
    else if(job.warnings?.length)notice(job.warnings.join(' '));
  }
  phases.forEach((phase,index)=>{const el=document.querySelector(`[data-phase="${phase}"]`);const current=phases.indexOf(job?.phase);el.classList.toggle('active',!!running&&current===index);el.classList.toggle('done',job?.status==='complete'||current>index);});
  if(job?.metadata){$('video-meta').textContent=`${job.metadata.width}×${job.metadata.height} / ${job.metadata.fps.toFixed(1)} FPS`;$('duration').textContent=time(job.metadata.duration);}
  if(job?.video){const url=fileURL(job.video);if($('player').getAttribute('src')!==url){$('player').src=url;$('player').hidden=false;$('empty-state').hidden=true;$('dropzone').removeAttribute('role');$('dropzone').removeAttribute('tabindex');}}
  const frames=job?.frames||[];
  if(frames.length&&!job?.video){$('empty-state').hidden=true;$('player').hidden=false;$('player').poster=fileURL(frames[0].image);}
  $('sample-count').textContent=`${frames.length} кадров`;
  if(frames.length!==lastFrameCount){
    $('filmstrip').replaceChildren();lastFrameCount=frames.length;
    frames.forEach((frame,index)=>{const tile=document.createElement('button');tile.className='frame-tile';tile.dataset.index=index;tile.title=`Кадр ${time(frame.time)}`;tile.setAttribute('aria-label',tile.title);tile.style.animationDelay=`${Math.min(index,12)*35}ms`;
      const image=document.createElement('img');image.src=fileURL(frame.image);image.loading='lazy';image.alt='';
      const label=document.createElement('span');label.className='frame-time';label.textContent=time(frame.time);
      const dot=document.createElement('span');dot.className='frame-dot';tile.append(image,label,dot);tile.onclick=()=>selectFrame(index,true);$('filmstrip').append(tile);
    });
  }
  document.querySelectorAll('.frame-tile').forEach(tile=>{const frame=frames[Number(tile.dataset.index)];tile.classList.toggle('suspicious',frame.spatial>=threshold);tile.classList.toggle('selected',Number(tile.dataset.index)===selected);});
  renderResult();renderDetail();renderSegments();drawChart();icons();
}

function renderResult(){
  const frames=job?.frames||[], clips=job?.clips||[], summary=job?.summary;
  const spatial=frames.filter(f=>Number.isFinite(f.spatial));
  const s=spatial.length ? spatial.reduce((sum,f)=>sum+f.spatial,0)/spatial.length:null;
  const t=clips.length ? clips.reduce((sum,c)=>sum+c.temporal,0)/clips.length:null;
  $('spatial-score').textContent=score(s);$('temporal-score').textContent=score(t);
  $('spatial-meter').style.width=`${(s||0)*100}%`;$('temporal-meter').style.width=`${(t||0)*100}%`;
  $('combined-row').hidden=!summary;$('combined-score').textContent=score(summary?.combined);
  $('result-coverage').hidden=!summary?.coverage;
  if(summary?.coverage){const c=summary.coverage;$('result-coverage').textContent=c.motion==='unsupported' ? `${summary.spatial_samples} кадров · движение недоступно` : `${summary.spatial_samples} кадров · клипы ${c.scored_clips}/${c.expected_clips}`;}
  const tag=$('result-tag');tag.className='tag';$('agreement').hidden=true;
  if(summary){tag.classList.add(summary.verdict);
    const verdicts={suspected:['ЕСТЬ СИГНАЛЫ','Возможное вмешательство',summary.reasons?.includes('cross_branch_intervals') ? 'Обе модели дали высокий балл на одном участке. Это повод проверить отрезок, не доказательство генерации.' : 'Сохранённый анализ выделил подозрительные интервалы. Это не подтверждение генерации.'],no_signals:['НЕТ СИГНАЛОВ','Сильных признаков не найдено','Оценки проверенных кадров и клипов низкие. Это не гарантирует подлинность видео.'],uncertain:['НЕОПРЕДЕЛЕННО','Нужна дополнительная проверка',(summary.reasons||[]).map(reason=>reviewReasons[reason]).filter(Boolean).join(' ')||'Данных недостаточно для более определённого заключения.']};
    const v=verdicts[summary.verdict];tag.textContent=v[0];$('verdict').textContent=v[1];$('verdict-detail').textContent=v[2];
    $('agreement').hidden=!summary.disagreement||summary.verdict==='uncertain';$('agreement').textContent='Оценки структуры и движения заметно расходятся.';
  }else if(job?.status==='error'){tag.textContent='ОШИБКА';$('verdict').textContent='Анализ не завершен';$('verdict-detail').textContent='Итоговое заключение отсутствует. Проверьте сообщение об ошибке.';
  }else if(job?.status==='cancelled'){tag.textContent='ОСТАНОВЛЕН';$('verdict').textContent='Анализ остановлен';$('verdict-detail').textContent='Частичные оценки не являются итоговым заключением.';
  }else if(job){tag.textContent='В РАБОТЕ';$('verdict').textContent='Исследуем видеоматериал';$('verdict-detail').textContent='Заключение появится после завершения обеих стадий анализа.';
  }else{tag.textContent='ОЖИДАНИЕ';$('verdict').textContent='Готов к анализу';$('verdict-detail').textContent='Две обученные модели подключаются при первом анализе.';}
}

function nearestClip(frame){return (job?.clips||[]).reduce((best,c)=>!best||Math.abs((c.start+c.end)/2-frame.time)<Math.abs((best.start+best.end)/2-frame.time)?c:best,null);}
function selectFrame(index,seek=false){selected=Math.max(0,Math.min(index,(job?.frames?.length||1)-1));renderDetail();document.querySelectorAll('.frame-tile').forEach(t=>t.classList.toggle('selected',Number(t.dataset.index)===selected));if(seek&&job?.video)$('player').currentTime=job.frames[selected].time;drawChart();}
function renderDetail(){
  const frame=job?.frames?.[selected];document.querySelectorAll('[data-view]').forEach(button=>button.classList.toggle('selected',button.dataset.view===view));
  $('detail-image').hidden=!frame;$('detail-empty').hidden=!!frame;$('detail-badge').hidden=!frame;
  if(!frame){$('selected-time').textContent='—';$('global-score').textContent='—';$('local-score').textContent='—';$('clip-score').textContent='—';$('detail-description').textContent='—';return;}
  const clip=nearestClip(frame);let image=frame.image,description=`Оценка кадра ${score(frame.spatial)}. Общий вид и пять локальных участков.`,badge='ИСХОДНЫЙ КАДР';
  if(view==='heatmap'){
    if(frame.heatmap){image=frame.heatmap;description=frame.heat_valid ? 'Grad-CAM полного кадра: области влияния на fake-логит. Это не маска генерации и не объяснение причины.' : 'Карта влияния для этого кадра не содержит положительной активации.';badge='GRAD-CAM / СТРУКТУРА';}
    else{description='Карта влияния рассчитывается для четырех кадров с наибольшими оценками. Для этого кадра карта отсутствует.';badge='КАРТА НЕДОСТУПНА';}
  }else if(view==='flow'){
    if(clip){image=clip.flow;description=`Вход второй модели: поля движения, ${time(clip.start)}–${time(clip.end)}. Оценка ${score(clip.temporal)}. Цвет — направление, яркость — скорость.`;badge='ОПТИЧЕСКИЙ ПОТОК';}
    else{description='Оценка движения еще не получена либо видео не соответствует длительности/FPS модели.';badge='ПОТОК НЕДОСТУПЕН';}
  }
  $('detail-image').src=fileURL(image);$('detail-image').alt=badge;$('detail-badge').textContent=badge;$('detail-description').textContent=description;$('selected-time').textContent=time(frame.time);
  $('global-score').textContent=score(frame.global_score);$('local-score').textContent=score(frame.local_score);$('clip-score').textContent=score(clip?.temporal);
}

function renderSegments(){
  const regions=[...(job?.summary?.segments||[]),...(job?.summary?.review_segments||[]).map(region=>({...region,review:true}))].sort((a,b)=>a.start-b.start);
  const frames=job?.frames||[], clips=job?.clips||[];
  $('segment-count').textContent=regions.length;
  $('flag-count').textContent=job ? `Выше порога: ${frames.filter(f=>f.spatial>=threshold).length} кадр. / ${clips.filter(c=>c.temporal>=threshold).length} клип.`:'Нет результатов';
  const container=$('segments');container.replaceChildren();
  if(!regions.length){const p=document.createElement('p');p.className='muted';p.textContent=job?.summary?'В заключении нет выделенных интервалов':'Анализ еще не завершен';container.append(p);return;}
  for(const {start,end,evidence,review} of regions){
    const button=document.createElement('button');button.className='segment';
    if(review)button.classList.add('segment-review');
    const icon=document.createElement('i');icon.dataset.lucide=review?'circle-help':'flag';
    const label=document.createElement('span');label.textContent=`${start.toFixed(1)}–${end.toFixed(1)} с`;
    const basis=document.createElement('span');basis.textContent=(evidence||[]).map(code=>evidenceLabels[code]).filter(Boolean).join(' · ')||'Сохранённый отчёт';
    button.append(icon,label,basis);
    button.onclick=()=>{if(frames.length)selectFrame(frames.reduce((best,f,i)=>Math.abs(f.time-start)<Math.abs(frames[best].time-start)?i:best,0));if(job.video)$('player').currentTime=start;$('player').scrollIntoView({behavior:'smooth',block:'center'});};
    container.append(button);
  }
}

function drawChart(){
  const canvas=$('score-chart'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const w=rect.width,h=rect.height;
  for(let i=0;i<=4;i++){ctx.strokeStyle='#dfe6df';ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(0,4+(h-8)*i/4);ctx.lineTo(w,4+(h-8)*i/4);ctx.stroke();}
  const duration=job?.metadata?.duration||1;
  const X=t=>Math.max(0,Math.min(w,t/duration*w)),Y=v=>4+(1-v)*(h-8);
  ctx.fillStyle='#a36d221c';for(const region of job?.summary?.review_segments||[])ctx.fillRect(X(region.start),0,X(region.end)-X(region.start),h);
  ctx.fillStyle='#df564c20';for(const region of job?.summary?.segments||[])ctx.fillRect(X(region.start),0,X(region.end)-X(region.start),h);
  ctx.setLineDash([3,4]);ctx.strokeStyle='#da8f83';ctx.beginPath();ctx.moveTo(0,Y(threshold));ctx.lineTo(w,Y(threshold));ctx.stroke();ctx.setLineDash([]);
  const sets=[[(job?.frames||[]).filter(f=>Number.isFinite(f.spatial)).map(f=>[f.time,f.spatial]),'#167958'],[(job?.clips||[]).map(c=>[(c.start+c.end)/2,c.temporal]),'#4086bd']];
  $('chart-empty').hidden=sets.some(([values])=>values.length);
  for(const [values,color]of sets){ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.beginPath();values.forEach(([t,s],i)=>i?ctx.lineTo(X(t),Y(s)):ctx.moveTo(X(t),Y(s)));ctx.stroke();for(const[t,s]of values){ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(t),Y(s),1.8,0,Math.PI*2);ctx.fill();}}
  if(job?.frames?.[selected]){const x=X(job.frames[selected].time);ctx.strokeStyle='#222f28';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
}

function choose(){if(!busy())$('file-input').click();}
$('new-button').onclick=choose;$('choose-button').onclick=event=>{event.stopPropagation();choose();};
$('file-input').onchange=event=>{upload(event.target.files[0]);event.target.value='';};
$('dropzone').onclick=event=>{if(!$('empty-state').hidden&&event.target!==$('choose-button'))choose();};
$('dropzone').onkeydown=event=>{if(event.key==='Enter'&& !$('empty-state').hidden){event.preventDefault();choose();}};
for(const name of ['dragenter','dragover'])$('dropzone').addEventListener(name,event=>{event.preventDefault();$('dropzone').classList.add('dragover');});
for(const name of ['dragleave','drop'])$('dropzone').addEventListener(name,event=>{event.preventDefault();$('dropzone').classList.remove('dragover');if(name==='drop')upload(event.dataTransfer.files[0]);});
$('cancel-button').onclick=async()=>{if(uploadRequest){uploadRequest.abort();return;}if(job){try{await responseJSON(await fetch(`/api/jobs/${job.id}/cancel`,{method:'POST'}));$('process-status').textContent='Остановка…';}catch(error){notice(error.message);}}};
$('threshold').oninput=event=>{threshold=Number(event.target.value);$('threshold-value').textContent=threshold.toFixed(2);render();};
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;renderDetail();});
document.querySelectorAll('.close-dialog').forEach(button=>button.onclick=()=>button.closest('dialog').close());
$('models-button').onclick=()=>{$('models-dialog').showModal();};
$('model-status-button').onclick=$('models-button').onclick;
$('history-button').onclick=()=>{const id=localStorage.getItem('trace-last-job');if(busy()){notice('Текущий анализ еще выполняется.');return;}if(!id){notice('Сохраненных анализов пока нет.');return;}notice('');job={id};started=Date.now();lastFrameCount=-1;poll();};
$('mobile-history-button').onclick=$('history-button').onclick;
$('report-button').onclick=()=>{$('json-download').href=fileURL('report.json');$('csv-download').href=fileURL('frames.csv');$('report-dialog').showModal();};
$('player').ontimeupdate=()=>{if(!job?.frames?.length)return;const t=$('player').currentTime;const index=job.frames.reduce((best,f,i)=>Math.abs(f.time-t)<Math.abs(job.frames[best].time-t)?i:best,0);if(index!==selected)selectFrame(index);};
$('score-chart').onclick=event=>{if(!job?.frames?.length)return;const r=event.currentTarget.getBoundingClientRect(),t=(event.clientX-r.left)/r.width*job.metadata.duration;selectFrame(job.frames.reduce((best,f,i)=>Math.abs(f.time-t)<Math.abs(job.frames[best].time-t)?i:best,0),true);};
new ResizeObserver(drawChart).observe($('score-chart'));icons();health();drawChart();
