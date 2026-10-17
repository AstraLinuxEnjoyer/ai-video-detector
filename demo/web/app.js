/* Local-only interface; every plotted value comes from the inference service. */
const $ = id => document.getElementById(id);
let job = null, selected = 0, view = 'frame', threshold = .65, timer = null, started = 0;
let uploadRequest = null, uploading = false, lastFrameCount = -1;
let lastExperienceState = 'empty';
const reducedMotion = {get matches(){return !TraceMotion.enabled;}};
const phases = ['decode','spatial','temporal','explain'];
const names = {decode:'Извлекаем кадры',spatial:'Анализ структуры',temporal:'Анализ движения',explain:'Карты влияния',models:'Загрузка весов',complete:'Анализ завершен'};
const icons = () => window.lucide?.createIcons();
const time = seconds => {const s = Math.max(0, seconds || 0); return `${Math.floor(s/60).toString().padStart(2,'0')}:${Math.floor(s%60).toString().padStart(2,'0')}`;};
const score = value => Number.isFinite(value) ? value.toFixed(3) : '—';
const fileURL = name => `/api/jobs/${job.id}/files/${encodeURIComponent(name)}`;
const busy = () => uploading || job && ['running','queued'].includes(job.status);
function notice(message){ $('notice').textContent = message; $('notice').hidden = !message; }
function progress(value){const percent=Math.min(100,Math.max(0,value));$('progress-fill').style.transform=`scaleX(${percent/100})`;$('progress-number').textContent=`${Math.round(percent)}%`;$('progress-fill').parentElement.setAttribute('aria-valuenow',Math.round(percent));}
async function responseJSON(response){const data = await response.json(); if(!response.ok) throw new Error(data.detail || 'Ошибка сервера'); return data;}

async function health(){
  try{const data = await responseJSON(await fetch('/api/health')); const ready = data.status==='ready'&&data.ffmpeg;
    $('server-status').textContent = ready ? 'Локально' : 'Требуется настройка';
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
  TraceMotion.resetDisclosure($('analysis-details'));
  $('player').pause(); $('player').removeAttribute('src');$('player').load();$('player').hidden=true;
  $('empty-state').hidden=false;$('scan-overlay').hidden=true;progress(0);
  $('dropzone').setAttribute('role','group');
  $('report-button').disabled=true;$('filmstrip').replaceChildren();$('file-name').textContent='Новая проверка';
  $('video-meta').textContent='—';$('duration').textContent='—'; render();
}

function upload(file){
  if(!file) return;
  if(busy()){notice('Сначала завершите или остановите текущий анализ.');return;}
  if(file.size>400*1024*1024){notice('Максимальный размер: 400 МБ.');return;}
  reset();notice('');uploading=true;started=Date.now();$('file-name').textContent=file.name;
  $('process-status').textContent='Загрузка файла';$('cancel-button').hidden=false;
  $('new-button').disabled=true;$('choose-button').disabled=true;
  render();
  const form=new FormData();form.append('file',file);const request=new XMLHttpRequest();uploadRequest=request;
  request.open('POST','/api/jobs');request.responseType='json';
  request.upload.onprogress=event=>{if(event.lengthComputable){progress(event.loaded/event.total*100);$('process-status').textContent=`Загрузка файла · ${Math.round(event.loaded/event.total*100)}%`;}};
  request.onload=()=>{uploading=false;uploadRequest=null;$('new-button').disabled=false;$('choose-button').disabled=false;
    if(request.status!==202){finishUploadError(request.response?.detail || 'Не удалось загрузить видео');return;}
    job={id:request.response.id,status:'queued',phase:'models',frames:[],clips:[]};
    localStorage.setItem('trace-last-job',job.id);progress(0);poll();};
  request.onerror=()=>finishUploadError('Соединение с сервером прервано');
  request.onabort=()=>finishUploadError('Загрузка отменена');request.send(form);
}
function finishUploadError(message){uploading=false;uploadRequest=null;$('new-button').disabled=false;$('choose-button').disabled=false;$('cancel-button').hidden=true;notice(message);$('process-status').textContent=message;render();}

async function poll(){
  if(!job) return;
  try{job=await responseJSON(await fetch(`/api/jobs/${job.id}`));render();
    if(['queued','running'].includes(job.status))timer=setTimeout(poll,800);
  }catch(error){notice(`Связь с сервером: ${error.message}. Результат можно восстановить через «Последний анализ».`);$('scan-overlay').hidden=true;$('cancel-button').hidden=true;}
}

function render(){
  const running=!!busy();
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
  renderResult();renderDetail();renderSegments();renderExperience();drawChart();icons();
}

function renderExperience(){
  const state=uploading?'running':job?.status==='complete'?'complete':busy()?'running':job?.status==='error'?'error':job?.status==='cancelled'?'cancelled':'empty';
  document.body.dataset.state=state;
  document.body.dataset.verdict=job?.summary?.verdict||'';
  $('result-panel').hidden=state==='empty';
  $('process-panel').hidden=state!=='running';
  $('new-button').hidden=state==='empty';
  $('explore-button').disabled=job?.status!=='complete';
  $('result-facts').hidden=job?.status!=='complete';
  $('result-panel').setAttribute('aria-busy',state==='running');
  $('analysis-details').hidden=!(job?.frames?.length);
  $('main-segment-count').textContent=job?.summary?.segments?.length??'—';
  $('main-sample-count').textContent=job?.summary?`${job.summary.spatial_samples} / ${job.summary.temporal_samples}`:'—';
  $('workspace-note').textContent=state==='complete'?'Результат сохранён на этом компьютере':'Без отправки видео в облако';
  if(uploading)$('process-status').textContent='Загрузка файла';
  if(state===lastExperienceState)return;
  const copy={empty:['У каждого видео<br>есть <span class="trace-word">след<span class="title-period">.</span><svg class="word-trace" viewBox="0 0 260 10" aria-hidden="true"><path d="M1 7 H82 L89 2 L95 9 L103 4 H259"/></svg></span>','Найдите признаки генерации. Узнайте, что осталось за кадром.'],running:['Кадр за кадром.','Анализируем изображение и движение. Результат появится здесь.'],complete:['Проверка завершена.','Краткий вывод перед вами. Кадры, карты и оценки — в подробностях.'],error:['Давайте попробуем<br>ещё раз.','Не удалось завершить проверку. Выберите другое видео или повторите попытку.'],cancelled:['Вы управляете<br>проверкой.','Анализ остановлен. Можно загрузить другое видео.']};
  $('hero-title').innerHTML=copy[state][0];
  $('hero-description').textContent=copy[state][1];
  if(!reducedMotion.matches&&state!=='empty')$('result-panel').animate([{opacity:.4,transform:'translateY(8px)'},{opacity:1,transform:'translateY(0)'}],{duration:250,easing:'cubic-bezier(.23,1,.32,1)'});
  lastExperienceState=state;
}

function renderResult(){
  const frames=job?.frames||[], clips=job?.clips||[];
  const spatial=frames.filter(f=>Number.isFinite(f.spatial));
  const s=spatial.length ? spatial.reduce((sum,f)=>sum+f.spatial,0)/spatial.length:null;
  const t=clips.length ? clips.reduce((sum,c)=>sum+c.temporal,0)/clips.length:null;
  $('spatial-score').textContent=score(s);$('temporal-score').textContent=score(t);
  $('spatial-meter').style.transform=`scaleX(${s||0})`;$('temporal-meter').style.transform=`scaleX(${t||0})`;
  const tag=$('result-tag');tag.className='tag';$('agreement').hidden=true;
  if(job?.summary){const summary=job.summary;tag.classList.add(summary.verdict);
    const verdicts={suspected:['ЕСТЬ СИГНАЛЫ','Есть повод присмотреться','В видео есть интервалы с подозрительными признаками. Посмотрите их подробнее — это ещё не подтверждение генерации.'],no_signals:['НЕТ СИГНАЛОВ','Явных признаков не найдено','На проверенных кадрах и клипах оценки ниже порога. Это не гарантирует подлинность видео.'],uncertain:['НЕОПРЕДЕЛЁННО','Однозначного ответа нет','Данных недостаточно для уверенного вывода. Рекомендуем дополнительную проверку.']};
    const v=verdicts[summary.verdict];tag.textContent=v[0];$('verdict').textContent=v[1];$('verdict-detail').textContent=v[2];
    $('agreement').hidden=!summary.disagreement;$('agreement').textContent='Модели расходятся: оценки структуры и движения заметно отличаются.';
  }else if(job?.status==='error'){tag.textContent='ОШИБКА';$('verdict').textContent='Анализ не завершен';$('verdict-detail').textContent='Итоговое заключение отсутствует. Проверьте сообщение об ошибке.';
  }else if(job?.status==='cancelled'){tag.textContent='ОСТАНОВЛЕН';$('verdict').textContent='Анализ остановлен';$('verdict-detail').textContent='Частичные оценки не являются итоговым заключением.';
  }else if(job||uploading){tag.textContent='В РАБОТЕ';$('verdict').textContent='Исследуем видеоматериал';$('verdict-detail').textContent='Сопоставляем признаки изображения и движения. Проверка может занять несколько минут.';
  }else{tag.textContent='ОЖИДАНИЕ';$('verdict').textContent='Готов к анализу';$('verdict-detail').textContent='Две обученные модели подключаются при первом анализе.';}
}

function nearestClip(frame){return (job?.clips||[]).reduce((best,c)=>!best||Math.abs((c.start+c.end)/2-frame.time)<Math.abs((best.start+best.end)/2-frame.time)?c:best,null);}
function selectFrame(index,seek=false){selected=Math.max(0,Math.min(index,(job?.frames?.length||1)-1));renderDetail();document.querySelectorAll('.frame-tile').forEach(t=>t.classList.toggle('selected',Number(t.dataset.index)===selected));if(seek&&job?.video)$('player').currentTime=job.frames[selected].time;drawChart();}
function renderDetail(){
  const frame=job?.frames?.[selected];document.querySelectorAll('[data-view]').forEach(button=>{button.classList.toggle('selected',button.dataset.view===view);button.setAttribute('aria-pressed',button.dataset.view===view);});
  $('detail-image').hidden=!frame;$('detail-empty').hidden=!!frame;$('detail-badge').hidden=!frame;
  if(!frame){TraceMotion.clearImage($('detail-image'));$('selected-time').textContent='—';$('global-score').textContent='—';$('local-score').textContent='—';$('clip-score').textContent='—';$('detail-description').textContent='—';return;}
  const clip=nearestClip(frame);let image=frame.image,description=`Оценка кадра ${score(frame.spatial)}. Общий вид и пять локальных участков.`,badge='ИСХОДНЫЙ КАДР';
  if(view==='heatmap'){
    if(frame.heatmap){image=frame.heatmap;description=frame.heat_valid ? 'Grad-CAM полного кадра: области влияния на fake-логит. Это не маска генерации и не объяснение причины.' : 'Карта влияния для этого кадра не содержит положительной активации.';badge='GRAD-CAM / СТРУКТУРА';}
    else{description='Карта влияния рассчитывается для четырех кадров с наибольшими оценками. Для этого кадра карта отсутствует.';badge='КАРТА НЕДОСТУПНА';}
  }else if(view==='flow'){
    if(clip){image=clip.flow;description=`Вход второй модели: поля движения, ${time(clip.start)}–${time(clip.end)}. Оценка ${score(clip.temporal)}. Цвет — направление, яркость — скорость.`;badge='ОПТИЧЕСКИЙ ПОТОК';}
    else{description='Оценка движения еще не получена либо видео не соответствует длительности/FPS модели.';badge='ПОТОК НЕДОСТУПЕН';}
  }
  TraceMotion.swapImage($('detail-image'),fileURL(image),badge);$('detail-badge').textContent=badge;$('detail-description').textContent=description;$('selected-time').textContent=time(frame.time);
  $('global-score').textContent=score(frame.global_score);$('local-score').textContent=score(frame.local_score);$('clip-score').textContent=score(clip?.temporal);
}

function spans(){const duration=job?.metadata?.duration||0;const raw=(job?.frames||[]).filter(f=>f.spatial>=threshold).map(f=>[f.time,Math.min(duration,f.time+2)]).concat((job?.clips||[]).filter(c=>c.temporal>=threshold).map(c=>[c.start,c.end])).sort((a,b)=>a[0]-b[0]);const result=[];for(const [start,end]of raw){const last=result.at(-1);if(last&&start<=last[1]+.05)last[1]=Math.max(last[1],end);else result.push([start,end]);}return result;}
function renderSegments(){const regions=spans();$('segment-count').textContent=regions.length;$('flag-count').textContent=job ? `${regions.length} интервалов выше порога`:'Нет результатов';const container=$('segments');container.replaceChildren();if(!regions.length){const p=document.createElement('p');p.className='muted';p.textContent=job?.summary?'Интервалы выше выбранного порога не найдены':'Анализ еще не завершен';container.append(p);return;}regions.forEach(([start,end])=>{const b=document.createElement('button');b.className='segment';const icon=document.createElement('i');icon.dataset.lucide='flag';const label=document.createElement('span');label.textContent=`${time(start)} — ${time(end)}`;const length=document.createElement('span');length.textContent=`${(end-start).toFixed(1)} с`;b.append(icon,label,length);b.onclick=()=>{const frames=job.frames;selectFrame(frames.reduce((best,f,i)=>Math.abs(f.time-start)<Math.abs(frames[best].time-start)?i:best,0),true);TraceMotion.scrollTo($('player'));};container.append(b);});}

function drawChart(){
  const canvas=$('score-chart'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const w=rect.width,h=rect.height;
  for(let i=0;i<=4;i++){ctx.strokeStyle='#354039';ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(0,4+(h-8)*i/4);ctx.lineTo(w,4+(h-8)*i/4);ctx.stroke();}
  const duration=job?.metadata?.duration||1;
  const X=t=>Math.max(0,Math.min(w,t/duration*w)),Y=v=>4+(1-v)*(h-8);
  ctx.setLineDash([3,4]);ctx.strokeStyle='#d7a28a';ctx.beginPath();ctx.moveTo(0,Y(threshold));ctx.lineTo(w,Y(threshold));ctx.stroke();ctx.setLineDash([]);
  const sets=[[(job?.frames||[]).filter(f=>Number.isFinite(f.spatial)).map(f=>[f.time,f.spatial]),'#ccf68b'],[(job?.clips||[]).map(c=>[(c.start+c.end)/2,c.temporal]),'#aabcf1']];
  $('chart-empty').hidden=sets.some(([values])=>values.length);
  for(const [values,color]of sets){ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.beginPath();values.forEach(([t,s],i)=>i?ctx.lineTo(X(t),Y(s)):ctx.moveTo(X(t),Y(s)));ctx.stroke();for(const[t,s]of values){ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(t),Y(s),1.8,0,Math.PI*2);ctx.fill();}}
  if(job?.frames?.[selected]){const x=X(job.frames[selected].time);ctx.strokeStyle='#a5afaa';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
}

function choose(){if(!busy())$('file-input').click();}
$('new-button').onclick=choose;$('choose-button').onclick=event=>{event.stopPropagation();choose();};
$('file-input').onchange=event=>{upload(event.target.files[0]);event.target.value='';};
$('dropzone').onclick=event=>{if(!$('empty-state').hidden&&event.target!==$('choose-button'))choose();};
for(const name of ['dragenter','dragover'])$('dropzone').addEventListener(name,event=>{event.preventDefault();$('dropzone').classList.add('dragover');});
for(const name of ['dragleave','drop'])$('dropzone').addEventListener(name,event=>{event.preventDefault();$('dropzone').classList.remove('dragover');if(name==='drop')upload(event.dataTransfer.files[0]);});
$('cancel-button').onclick=async()=>{if(uploadRequest){uploadRequest.abort();return;}if(job){try{await responseJSON(await fetch(`/api/jobs/${job.id}/cancel`,{method:'POST'}));$('process-status').textContent='Остановка…';}catch(error){notice(error.message);}}};
$('threshold').oninput=event=>{threshold=Number(event.target.value);$('threshold-value').textContent=threshold.toFixed(2);render();};
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;renderDetail();});
document.querySelectorAll('.close-dialog').forEach(button=>button.onclick=event=>TraceMotion.closeDialog(button.closest('dialog'),{immediate:event.detail===0}));
$('models-button').onclick=()=>{TraceMotion.openDialog($('models-dialog'));};
$('model-status-button').onclick=$('models-button').onclick;
$('history-button').onclick=()=>{const id=localStorage.getItem('trace-last-job');if(busy()){notice('Текущий анализ еще выполняется.');return;}if(!id){notice('Сохраненных анализов пока нет.');return;}notice('');job={id};started=Date.now();lastFrameCount=-1;poll();};
$('report-button').onclick=()=>{$('json-download').href=fileURL('report.json');$('csv-download').href=fileURL('frames.csv');TraceMotion.openDialog($('report-dialog'));};
$('player').ontimeupdate=()=>{if(!job?.frames?.length)return;const t=$('player').currentTime;const index=job.frames.reduce((best,f,i)=>Math.abs(f.time-t)<Math.abs(job.frames[best].time-t)?i:best,0);if(index!==selected)selectFrame(index);};
$('score-chart').onclick=event=>{if(!job?.frames?.length)return;const r=event.currentTarget.getBoundingClientRect(),t=(event.clientX-r.left)/r.width*job.metadata.duration;selectFrame(job.frames.reduce((best,f,i)=>Math.abs(f.time-t)<Math.abs(job.frames[best].time-t)?i:best,0),true);};
TraceMotion.mount();
document.addEventListener('trace:disclosure-settled',drawChart);
$('explore-button').onclick=()=>{
  const details=$('analysis-details');
  if(!details.open)TraceMotion.disclose(details,true);
  details.querySelector('summary').focus({preventScroll:true});
  requestAnimationFrame(()=>requestAnimationFrame(()=>TraceMotion.scrollTo(details)));
};
document.querySelectorAll('dialog').forEach(dialog=>dialog.addEventListener('click',event=>{
  if(event.target!==dialog)return;
  const r=dialog.getBoundingClientRect();
  if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)TraceMotion.closeDialog(dialog);
}));
new ResizeObserver(drawChart).observe($('score-chart'));icons();health();render();
