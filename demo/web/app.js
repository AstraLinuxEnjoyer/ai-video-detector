/* Local-only interface; every plotted value comes from the inference service. */
const $ = id => document.getElementById(id);
let job = null, selected = 0, view = 'spatial', detailIndex = 0, timer = null, started = 0;
let uploadRequest = null, uploading = false, lastFrameCount = -1;
let lastExperienceState = 'empty';
const reducedMotion = {get matches(){return !TraceMotion.enabled;}};
const phases = ['decode','spatial','temporal','spectral','explain'];
const names = {decode:'Извлекаем кадры',spatial:'Анализ структуры',temporal:'Анализ движения',spectral:'Анализ спектра',explain:'Карты влияния',models:'Загрузка весов',complete:'Анализ завершен'};
const fixedThreshold = () => job?.summary?.threshold ?? .65;
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
    for(const [kind,title] of [['spatial','EfficientNet V2-S / hybrid'],['temporal','R(2+1)D-18 / motion flow'],['spectral','Spectral / face v3']]){
      const row = document.createElement('div'); row.className = 'model-info-row';
      const strong=document.createElement('strong'); strong.textContent=title;
      const description=document.createElement('span'); description.textContent=data.models?.[kind] ? (kind==='spectral'?'Вспомогательный сигнал':`Лучшие веса · эпоха ${data.models[kind].epoch}`) : 'Веса отсутствуют';
      const hash=document.createElement('code'); hash.textContent=data.models?.[kind]?.sha256 || '';
      row.append(strong,description,hash); info.append(row);
    }
  }catch(error){$('server-status').textContent='Нет подключения';$('server-dot').className='status-dot error';notice(error.message);}
}

function reset(){
  clearTimeout(timer); job=null;selected=0;view='spatial';detailIndex=0;lastFrameCount=-1;
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
  document.querySelectorAll('.frame-tile').forEach(tile=>{const frame=frames[Number(tile.dataset.index)];tile.classList.toggle('suspicious',frame.spatial>=fixedThreshold());tile.classList.toggle('selected',Number(tile.dataset.index)===selected);});
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
  $('result-panel').setAttribute('aria-busy',state==='running');
  $('analysis-details').hidden=state!=='complete';
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
  const tag=$('result-tag');tag.className='tag';
  if(job?.summary){const summary=job.summary;tag.classList.add(summary.verdict);
    const verdicts={suspected:['ЕСТЬ СИГНАЛЫ','Есть повод присмотреться','В видео есть интервалы с подозрительными признаками. Посмотрите их подробнее — это ещё не подтверждение генерации.'],no_signals:['НЕТ СИГНАЛОВ','Явных признаков не найдено','На проверенных кадрах и клипах оценки ниже порога. Это не гарантирует подлинность видео.'],uncertain:['НЕОПРЕДЕЛЁННО','Однозначного ответа нет','Данных недостаточно для уверенного вывода. Рекомендуем дополнительную проверку.']};
    const v=verdicts[summary.verdict];tag.textContent=v[0];$('verdict').textContent=v[1];$('verdict-detail').textContent=v[2];
    $('report-summary').textContent=`Общий балл ${score(summary.combined)} · ${summary.segments.length} согласованных интервалов`;
  }else if(job?.status==='error'){tag.textContent='ОШИБКА';$('verdict').textContent='Анализ не завершен';$('verdict-detail').textContent='Итоговое заключение отсутствует. Проверьте сообщение об ошибке.';
  }else if(job?.status==='cancelled'){tag.textContent='ОСТАНОВЛЕН';$('verdict').textContent='Анализ остановлен';$('verdict-detail').textContent='Частичные оценки не являются итоговым заключением.';
  }else if(job||uploading){tag.textContent='В РАБОТЕ';$('verdict').textContent='Исследуем видеоматериал';$('verdict-detail').textContent='Сопоставляем признаки изображения и движения. Проверка может занять несколько минут.';
  }else{tag.textContent='ОЖИДАНИЕ';$('verdict').textContent='Готов к анализу';$('verdict-detail').textContent='Две обученные модели подключаются при первом анализе.';}
}

const nearest = (items, timestamp, at) => items.reduce((best,item,i)=>Math.abs(at(item)-timestamp)<Math.abs(at(items[best])-timestamp)?i:best,0);
const clipTime = c => (c.start+c.end)/2;
const timestamp = t => `${t.toFixed(2)} с`;
const reasons = {cross_branch_intervals:'Совпадение кадра и движения',single_branch_persistence:'Повтор одной ветки',isolated_high_scores:'Отдельный высокий отсчёт',elevated_mean_score:'Повышенный средний балл',branch_disagreement:'Оценки расходятся',motion_not_complete:'Неполная оценка движения',spectral_review:'Повышенный сигнал спектра',low_scores_with_complete_motion:'Нет сильных сигналов'};
function selectFrame(index,seek=false){
  selected=Math.max(0,Math.min(index,(job?.frames?.length||1)-1));
  if(seek){view='spatial';if(job?.video)$('player').currentTime=job.frames[selected].time;}
  renderDetail();document.querySelectorAll('.frame-tile').forEach(t=>t.classList.toggle('selected',Number(t.dataset.index)===selected));drawChart();
}

function reportPicture(image, empty, file, label, fallback){
  const target=$(image);
  target.hidden=!file;$(empty).hidden=!!file;$(empty).textContent=fallback;
  if(file)TraceMotion.swapImage(target,fileURL(file),label);else TraceMotion.clearImage(target);
}

function drawSpectrum(sample){
  const svg=$('spectrum-chart');svg.replaceChildren();svg.toggleAttribute('hidden',!sample);
  if(!sample)return;
  function element(tag, attrs, text){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);if(text)node.textContent=text;svg.append(node);}
  for(const v of [0,-20,-40,-60,-80]){const y=18-v*1.8;element('line',{x1:44,x2:390,y1:y,y2:y,stroke:'#49534d'});element('text',{x:7,y:y+4},`${v}`);}
  sample.bands_db.forEach((v,i)=>{const h=Math.max(0,Math.min(144,(v+80)*1.8));element('rect',{x:47+i*34,y:162-h,width:28,height:h,fill:i<5?'#ccf68b':'#aabcf1'});});
  element('text',{x:47,y:184},'Низкие частоты');element('text',{x:390,y:184,'text-anchor':'end'},'Высокие частоты');
}

function renderDetail(){
  const frames=job?.frames||[], clips=job?.clips||[], spec=job?.spectral, summary=job?.summary;
  const list=view==='spectral'?(spec?.samples||[]):view==='temporal'?clips:frames;
  const index=Math.min(view==='spatial'?selected:detailIndex,Math.max(0,list.length-1));
  const row=list[index], picker=$('sample-picker');
  document.querySelectorAll('[data-view]').forEach(button=>{button.classList.toggle('selected',button.dataset.view===view);button.setAttribute('aria-pressed',button.dataset.view===view);});
  TraceMotion.syncTabs();
  const labels=list.map(item=>view==='temporal'?`${timestamp(item.start)} – ${timestamp(item.end)} · ${score(item.temporal)}`:
    `${timestamp(item.time)}${view==='spatial'?` · ${score(item.spatial)}${item.heat_valid?' · Grad-CAM':''}`:''}`);
  if(picker.dataset.signature!==labels.join('|')){picker.replaceChildren(...labels.map((label,i)=>new Option(label,i)));picker.dataset.signature=labels.join('|');}
  picker.disabled=!row;picker.value=row?String(index):'';
  const frame=view==='temporal'&&row&&frames.length?frames[nearest(frames,clipTime(row),f=>f.time)]:view==='spatial'?row:null;
  const sample=view==='spectral'?row:null;
  const left=sample?.image||frame?.image, right=view==='spatial'?(frame?.heat_valid?frame.heatmap:null):view==='temporal'?row?.flow:null;
  $('detail-caption').textContent=view==='spectral'?'Канал яркости и область лица':view==='temporal'?'Ближайший кадр к середине клипа':'Исходный кадр';
  $('evidence-caption').textContent=view==='spectral'?'Энергия частотных полос, дБ':view==='temporal'?'Оптический поток':'Карта влияния Grad-CAM';
  reportPicture('detail-image','detail-empty',left,$('detail-caption').textContent,'Нет данных для этого отсчёта');
  reportPicture('evidence-image','evidence-empty',right,$('evidence-caption').textContent,view==='spatial'?'Для этого кадра карта не рассчитана':'Оценка недоступна');
  drawSpectrum(sample);if(sample)$('evidence-empty').hidden=true;
  $('selected-time').textContent=row?timestamp(view==='temporal'?row.start:row.time):'—';
  $('detail-description').textContent=view==='spatial'?'Grad-CAM отмечает области, влияющие на балл полного кадра. Это не маска подмены.':view==='temporal'?'Цвет показывает направление, яркость — величину смещения. Это вход модели, а не карта её внимания.':'График показывает входной спектр выбранного кадра. Рамка указывает область лица, не вмешательство. Балл спектра рассчитан на весь ролик.';
  const stats=[['Общий балл',score(summary?.combined)],['Средний балл ветки',score(summary?.[view])],['Вес в общем балле',Number.isFinite(summary?.weights?.[view])?`${Math.round(summary.weights[view]*100)}%`:'—']];
  if(view==='spatial')stats.push(['Оценка кадра',score(frame?.spatial)],['Общий вид',score(frame?.global_score)],['Локальные участки',score(frame?.local_score)],['Кадры',String(frames.length)]);
  if(view==='temporal')stats.push(['Оценка клипа',score(row?.temporal)],['Клипы',`${clips.length} / ${summary?.coverage?.expected_clips??clips.length}`]);
  if(view==='spectral')stats.push(['Порог спектра',score(spec?.threshold)],['Лицо в кадрах',`${spec?.face_frames??0} / ${list.length}`],['Только спектр',score(spec?.variants?.spectrum?.score)],['Шумовой остаток',score(spec?.variants?.noise?.score)],['Только лицо',score(spec?.variants?.face?.score)]);
  if(summary?.reasons?.length)stats.push(['Основание',summary.reasons.map(r=>reasons[r]).filter(Boolean).join(' · ')]);
  $('detail-stats').replaceChildren(...stats.map(([name,value])=>{const block=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=name;dd.textContent=value;block.append(dt,dd);return block;}));
}

function renderSegments(){
  const summary=job?.summary, regions=[...(summary?.segments||[]),...(summary?.review_segments||[]).map(r=>({...r,review:true}))].sort((a,b)=>a.start-b.start);
  const container=$('segments');container.replaceChildren();$('segment-count').textContent=regions.length;
  if(!regions.length){const p=document.createElement('p');p.className='muted';p.textContent='Нет выделенных интервалов';container.append(p);return;}
  for(const {start,end,review} of regions){const button=document.createElement('button'),icon=document.createElement('i'),label=document.createElement('span'),note=document.createElement('span');
    button.className='segment';icon.dataset.lucide=review?'circle-help':'flag';label.textContent=`${timestamp(start)} – ${timestamp(end)}`;note.textContent=review?'Ручная проверка':'Согласованный сигнал';button.append(icon,label,note);
    button.onclick=()=>{if(job.video)$('player').currentTime=start;selectFrame(nearest(job.frames,start,f=>f.time));TraceMotion.scrollTo($('player'));};container.append(button);
  }
}

function drawChart(){
  const canvas=$('score-chart'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const w=rect.width,h=rect.height;
  for(let i=0;i<=4;i++){ctx.strokeStyle='#354039';ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(0,4+(h-8)*i/4);ctx.lineTo(w,4+(h-8)*i/4);ctx.stroke();}
  const duration=job?.metadata?.duration||1;
  const X=t=>Math.max(0,Math.min(w,t/duration*w)),Y=v=>4+(1-v)*(h-8);
  for(const [regions,color] of [[job?.summary?.review_segments,'#a97b3626'],[job?.summary?.segments,'#d88a6938']]){ctx.fillStyle=color;for(const r of regions||[])ctx.fillRect(X(r.start),0,X(r.end)-X(r.start),h);}
  const threshold=fixedThreshold();$('threshold-label').textContent=`Порог ${score(threshold)}`;
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
$('sample-picker').onchange=event=>{const index=Number(event.target.value);if(view==='spatial')selectFrame(index);else{detailIndex=index;renderDetail();}};
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;const t=job?.frames?.[selected]?.time||0;detailIndex=view==='temporal'?nearest(job?.clips||[],t,clipTime):nearest(job?.spectral?.samples||[],t,s=>s.time);renderDetail();});
document.querySelectorAll('.close-dialog').forEach(button=>button.onclick=event=>TraceMotion.closeDialog(button.closest('dialog'),{immediate:event.detail===0}));
$('models-button').onclick=()=>{TraceMotion.openDialog($('models-dialog'));};
$('model-status-button').onclick=$('models-button').onclick;
$('history-button').onclick=()=>{const id=localStorage.getItem('trace-last-job');if(busy()){notice('Текущий анализ еще выполняется.');return;}if(!id){notice('Сохраненных анализов пока нет.');return;}notice('');job={id};started=Date.now();lastFrameCount=-1;poll();};
$('report-button').onclick=()=>{$('html-download').hidden=!job.human_report;if(job.human_report)$('html-download').href=fileURL(job.human_report);$('json-download').href=fileURL('report.json');$('csv-download').href=fileURL('frames.csv');TraceMotion.openDialog($('report-dialog'));};
$('player').ontimeupdate=()=>{if(!job?.frames?.length)return;const t=$('player').currentTime;const index=job.frames.reduce((best,f,i)=>Math.abs(f.time-t)<Math.abs(job.frames[best].time-t)?i:best,0);if(index!==selected)selectFrame(index);};
$('score-chart').onclick=event=>{if(!job?.frames?.length)return;const r=event.currentTarget.getBoundingClientRect(),t=(event.clientX-r.left)/r.width*job.metadata.duration;selectFrame(job.frames.reduce((best,f,i)=>Math.abs(f.time-t)<Math.abs(job.frames[best].time-t)?i:best,0),true);};
TraceMotion.mount();
document.addEventListener('trace:disclosure-settled',drawChart);
$('explore-button').onclick=()=>{
  const details=$('analysis-details');
  view='spatial';const frames=job.frames||[];const top=frames.reduce((best,f,i)=>f.heat_valid&&(best<0||f.spatial>frames[best].spatial)?i:best,-1);if(top>=0)selectFrame(top);else renderDetail();
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
