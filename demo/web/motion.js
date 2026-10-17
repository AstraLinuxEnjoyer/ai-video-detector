/* TRACE interaction layer. Native controls remain usable when motion is off. */
window.TraceMotion = (() => {
  'use strict';
  const root = document.documentElement;
  const systemMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  const easeOut = 'cubic-bezier(.23,1,.32,1)';
  let preference;
  try { preference = localStorage.getItem('trace-motion'); } catch {}
  let enabled = preference ? preference === 'full' : !systemMotion.matches;
  let lenis = null;
  let scrollWaitFrame = 0;
  const disclosures = new WeakMap();
  const dialogs = new WeakMap();
  const imageChanges = new WeakMap();
  const tabGroups = new Set();
  const activeAnimations = new Set();

  function animate(element, frames, options) {
    const animation = element.animate(frames, options);
    activeAnimations.add(animation);
    const release = () => activeAnimations.delete(animation);
    animation.addEventListener('finish', release, {once:true});
    animation.addEventListener('cancel', release, {once:true});
    return animation;
  }

  function configureScrolling() {
    cancelAnimationFrame(scrollWaitFrame);
    lenis?.destroy();
    lenis = null;
    if (enabled && window.Lenis) {
      lenis = new Lenis({
        autoRaf:true,
        smoothWheel:true,
        lerp:0.1,
        syncTouch:false,
        prevent:node => node instanceof Element && !!node.closest('[data-lenis-prevent], dialog'),
      });
      if (document.querySelector('dialog[open]')) lenis.stop();
    }
  }

  function updateMotion() {
    root.dataset.motion = enabled ? 'full' : 'reduced';
    const button = document.getElementById('motion-toggle');
    if (button) {
      button.setAttribute('aria-pressed', String(enabled));
      button.querySelector('span').textContent = enabled ? 'Анимации включены' : 'Анимации выключены';
      button.title = enabled ? 'Выключить анимации' : 'Включить анимации, включая плавную прокрутку';
    }
    if (!enabled) {
      for (const details of document.querySelectorAll('details')) {
        const record = disclosures.get(details);
        if (record) finishDisclosure(details, record.opening);
      }
      for (const animation of [...activeAnimations]) animation.finish();
      document.querySelectorAll('.reveal-ready').forEach(element => element.classList.add('is-revealed'));
    }
    configureScrolling();
    syncTabs();
    document.dispatchEvent(new CustomEvent('trace:motion-change', {detail:{enabled}}));
  }

  function scrollTo(target, {offset = 28, immediate = false} = {}) {
    if (!target) return;
    cancelAnimationFrame(scrollWaitFrame);
    const move = () => {
      scrollWaitFrame=0;
      const header = document.querySelector('.topbar');
      const topOffset = (header?.getBoundingClientRect().height || 0) + offset;
      const top = typeof target === 'number' ? target : target.getBoundingClientRect().top + scrollY - topOffset;
      if (lenis && !immediate) {
        lenis.resize();
        // Wait until the expanding report creates enough scrollable space. Otherwise
        // Lenis clamps the destination to the old page height and stops short.
        if (typeof target!=='number' && disclosures.get(target)?.opening && top>lenis.limit+1) {
          scrollWaitFrame=requestAnimationFrame(move);return;
        }
        // A coordinate avoids applying the CSS scroll-padding a second time.
        lenis.scrollTo(top, {duration:0.85});
      } else {
        window.scrollTo({top, behavior:enabled && !immediate ? 'smooth' : 'instant'});
      }
    };
    move();
  }

  function finishDisclosure(details, opening) {
    const record = disclosures.get(details);
    disclosures.delete(details);
    record?.animation.cancel();
    record?.contentAnimations.forEach(animation => animation.cancel());
    details.open = opening;
    details.dataset.expanded = String(opening);
    details.querySelector(':scope > summary').setAttribute('aria-expanded', String(opening));
    details.style.removeProperty('height');
    details.style.removeProperty('overflow');
    const body = details.querySelector(':scope > .disclosure-body');
    if (body) body.inert = !opening;
    lenis?.resize();
    syncTabs();
    details.dispatchEvent(new CustomEvent('trace:disclosure-settled', {bubbles:true}));
  }

  function disclose(details, opening, {immediate = false} = {}) {
    const previous = disclosures.get(details);
    const start = details.getBoundingClientRect().height;
    const body = details.querySelector(':scope > .disclosure-body');
    const bodyOpacity = body ? getComputedStyle(body).opacity : '1';
    const bodyTransform = body ? getComputedStyle(body).transform : 'none';
    previous?.animation.cancel();
    previous?.contentAnimations.forEach(animation => animation.cancel());
    disclosures.delete(details);
    if (!enabled || immediate) { finishDisclosure(details, opening); return; }

    // Keep native content laid out until exit finishes; measure both real endpoints.
    details.open = true;
    if (body) body.inert = !opening;
    details.dataset.expanded = String(opening);
    const summary = details.querySelector(':scope > summary');
    summary.setAttribute('aria-expanded', String(opening));
    const style = getComputedStyle(details);
    const inset = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth)
      + parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    const end = opening ? details.getBoundingClientRect().height : summary.getBoundingClientRect().height + inset;
    details.style.overflow = 'clip';
    const duration = details.id === 'analysis-details' ? 420 : 280;
    if (!opening && lenis && details.id === 'analysis-details') {
      const fullHeight=details.getBoundingClientRect().height;
      const futureLimit=Math.max(0,document.documentElement.scrollHeight-(fullHeight-end)-innerHeight);
      if(scrollY>futureLimit)lenis.scrollTo(futureLimit,{duration:duration/1000});
    }
    const animation = animate(details, [{height:`${start}px`}, {height:`${end}px`}], {duration,easing:easeOut});
    const contentAnimations = [];
    if (body) {
      contentAnimations.push(animate(body, [
        {opacity:previous ? bodyOpacity : opening ? 0.25 : 1, transform:previous ? bodyTransform : opening ? 'translateY(12px)' : 'translateY(0)'},
        {opacity:opening ? 1 : 0, transform:opening ? 'translateY(0)' : 'translateY(8px)'},
      ], {duration:opening ? 280 : 180, easing:easeOut,fill:'both'}));
    }
    disclosures.set(details,{animation,contentAnimations,opening});
    animation.onfinish = () => finishDisclosure(details, opening);
  }

  function resetDisclosure(details) { finishDisclosure(details, false); }

  function setupDisclosures() {
    document.querySelectorAll('details').forEach(details => {
      const summary = details.querySelector(':scope > summary');
      const body = document.createElement('div');
      body.className = 'disclosure-body';
      while (summary.nextSibling) body.append(summary.nextSibling);
      details.append(body);
      body.inert = !details.open;
      details.dataset.expanded = String(details.open);
      summary.setAttribute('aria-expanded', String(details.open));
      summary.addEventListener('click', event => {
        event.preventDefault();
        const opening = !(disclosures.get(details)?.opening ?? details.open);
        disclose(details, opening, {immediate:event.detail === 0});
      });
      details.addEventListener('toggle', () => {
        if (disclosures.has(details)) return;
        body.inert = !details.open;
        details.dataset.expanded = String(details.open);
        summary.setAttribute('aria-expanded', String(details.open));
        lenis?.resize();
        syncTabs();
      });
    });
  }

  function openDialog(dialog) {
    const previous = dialogs.get(dialog);
    const from = previous ? getComputedStyle(dialog) : null;
    const opacity = from?.opacity ?? 0;
    const transform = from?.transform ?? 'translateY(16px) scale(.96)';
    previous?.cancel();
    dialogs.delete(dialog);
    delete dialog.dataset.closing;
    if (!dialog.open) dialog.showModal();
    lenis?.stop();
    if (!enabled) return;
    const animation = animate(dialog,[{opacity,transform},{opacity:1,transform:'translateY(0) scale(1)'}],{duration:280,easing:easeOut});
    dialogs.set(dialog,animation);
    animation.onfinish = () => dialogs.delete(dialog);
  }

  function closeDialog(dialog, {immediate = false} = {}) {
    if (!dialog?.open) return;
    if (!enabled || immediate) {dialogs.get(dialog)?.cancel();dialog.close();return;}
    if (dialog.dataset.closing) return;
    const current = getComputedStyle(dialog);
    const from = {opacity:current.opacity,transform:current.transform};
    dialogs.get(dialog)?.cancel();
    dialog.dataset.closing = 'true';
    const animation = animate(dialog,[from,{opacity:0,transform:'translateY(12px) scale(.96)'}],{duration:220,easing:easeOut});
    dialogs.set(dialog,animation);
    animation.onfinish = () => {dialog.close();dialogs.delete(dialog);delete dialog.dataset.closing;};
  }

  function syncTabs() {
    tabGroups.forEach(group => {
      const selected = group.querySelector('button.selected');
      const glider = group.querySelector('.tab-glider');
      if (!selected || !selected.offsetWidth) return;
      glider.style.transform = `translateX(${selected.offsetLeft}px) scaleX(${selected.offsetWidth})`;
      glider.style.height = `${selected.offsetHeight}px`;
      glider.style.top = `${selected.offsetTop}px`;
      if (!group.classList.contains('tabs-ready')) requestAnimationFrame(() => group.classList.add('tabs-ready'));
    });
  }

  function clearImage(image) {
    const previous = imageChanges.get(image);
    previous?.animations?.forEach(animation => animation.cancel());
    previous?.ghost?.remove();
    imageChanges.delete(image);
  }

  function swapImage(image, source, alt) {
    if (image.getAttribute('src') === source || imageChanges.get(image)?.source === source) { image.alt = alt; return; }
    clearImage(image);
    if (!enabled || !image.getAttribute('src') || !image.complete || !image.naturalWidth) {
      image.src = source; image.alt = alt; return;
    }
    const request = {source,animations:[]};
    imageChanges.set(image, request);
    const preload = new Image();
    preload.onload = () => {
      if (imageChanges.get(image) !== request) return;
      if (!enabled) {image.src=source;image.alt=alt;clearImage(image);return;}
      const ghost = image.cloneNode();
      ghost.removeAttribute('id'); ghost.alt = ''; ghost.setAttribute('aria-hidden','true');
      ghost.className = 'image-transition-layer';
      image.parentElement.append(ghost);request.ghost = ghost;
      image.src = source; image.alt = alt;
      request.animations.push(animate(image,[{opacity:0},{opacity:1}],{duration:250,easing:easeOut}));
      const exit = animate(ghost,[{opacity:1},{opacity:0}],{duration:250,easing:easeOut});
      request.animations.push(exit);
      exit.onfinish = () => {ghost.remove();if(imageChanges.get(image)===request)imageChanges.delete(image);};
    };
    preload.onerror = () => {if(imageChanges.get(image)===request){image.src=source;image.alt=alt;imageChanges.delete(image);}};
    preload.src = source;
  }

  function setupPointer() {
    const stage = document.getElementById('empty-state');
    const scene = stage.querySelector('.frame-scene');
    const halo = document.createElement('span');halo.className = 'pointer-halo';halo.setAttribute('aria-hidden','true');
    const reticle = document.createElement('span');reticle.className = 'pointer-reticle';reticle.setAttribute('aria-hidden','true');
    stage.prepend(halo,reticle);
    let frame=0,point=null;
    stage.addEventListener('pointermove', event => {
      if (!enabled || !finePointer.matches || event.pointerType === 'touch') return;
      point={x:event.clientX,y:event.clientY};
      if (frame) return;
      frame=requestAnimationFrame(() => {
        frame=0;
        const rect=stage.getBoundingClientRect(),x=point.x-rect.left,y=point.y-rect.top;
        stage.classList.add('pointer-active');
        halo.style.transform=`translate3d(${x-220}px,${y-220}px,0)`;
        reticle.style.transform=`translate3d(${x-18}px,${y-18}px,0)`;
        scene.style.transform=`translate3d(${(x/rect.width-.5)*20}px,${(y/rect.height-.5)*12}px,0) rotateX(${(y/rect.height-.5)*-14}deg) rotateY(${(x/rect.width-.5)*22}deg)`;
      });
    });
    const reset=()=>{cancelAnimationFrame(frame);frame=0;stage.classList.remove('pointer-active');scene.style.removeProperty('transform');};
    stage.addEventListener('pointerleave', reset);
    document.addEventListener('trace:motion-change', reset);
    new IntersectionObserver(entries=>{
      const visible=entries[0].isIntersecting;
      stage.classList.toggle('specimen-paused',!visible);
      if(!visible)reset();
    }).observe(stage);
  }

  function setupScroll() {
    const header=document.querySelector('.topbar');
    const nav=header.querySelector('.topnav'),navInk=document.createElement('span');
    navInk.className='nav-indicator';navInk.setAttribute('aria-hidden','true');nav.append(navInk);
    let currentNav='';
    const moveNav=(hash,force=false)=>{
      if(currentNav===hash&&!force)return;
      currentNav=hash;
      nav.querySelectorAll('a').forEach(link=>{
        const selected=link.getAttribute('href')===hash;
        link.classList.toggle('current',selected);
        if(selected){link.setAttribute('aria-current','location');navInk.style.transform=`translateX(${link.offsetLeft}px) scaleX(${Math.max(16,link.offsetWidth*.3)})`;}
        else link.removeAttribute('aria-current');
      });
    };
    const meter=document.createElement('span');meter.className='reading-progress';meter.setAttribute('aria-hidden','true');
    document.body.append(meter);
    let scrollFrame=0;
    const update=()=>{
      scrollFrame=0;
      const total=document.documentElement.scrollHeight-innerHeight;
      meter.style.transform=`scaleX(${total>0?Math.min(1,scrollY/total):0})`;
      header.classList.toggle('is-scrolled',scrollY>24);
      moveNav(document.getElementById('method').getBoundingClientRect().top<header.offsetHeight+160?'#method':'#workspace');
    };
    addEventListener('scroll',()=>{if(!scrollFrame)scrollFrame=requestAnimationFrame(update);},{passive:true});
    for(const event of ['wheel','touchstart','keydown'])addEventListener(event,()=>{cancelAnimationFrame(scrollWaitFrame);scrollWaitFrame=0;},{passive:true});
    update();
    addEventListener('resize',()=>{update();moveNav(currentNav,true);},{passive:true});

    // Only supplementary explanation enters on scroll; task controls stay visible.
    const reveals=[document.querySelector('.method-intro'),...document.querySelectorAll('.method-explanations > details'),document.querySelector('footer')];
    const observer=new IntersectionObserver(entries=>{
      entries.forEach(entry=>{
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-revealed');observer.unobserve(entry.target);
      });
    },{threshold:0.12,rootMargin:'0px 0px -20px 0px'});
    reveals.forEach((element,index)=>{
      if(!element)return;
      element.classList.add('reveal-ready');
      element.style.transitionDelay=`${index>0&&index<4?(index-1)*60:0}ms`;
      if(!enabled)element.classList.add('is-revealed');
      observer.observe(element);
    });
    document.addEventListener('focusin',event=>event.target.closest('.reveal-ready')?.classList.add('is-revealed'));
    document.querySelectorAll('a[href^="#"]').forEach(anchor=>anchor.addEventListener('click',event=>{
      const target=document.querySelector(anchor.getAttribute('href'));
      if(!target||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
      event.preventDefault();
      scrollTo(target,{immediate:event.detail===0});
      if(event.detail===0){target.setAttribute('tabindex','-1');target.focus({preventScroll:true});target.addEventListener('blur',()=>target.removeAttribute('tabindex'),{once:true});}
      history.replaceState(null,'',anchor.getAttribute('href'));
    }));
  }

  function mount() {
    setupDisclosures();
    setupPointer();
    setupScroll();
    document.querySelectorAll('.segmented').forEach(group=>{
      const glider=document.createElement('span');glider.className='tab-glider';glider.setAttribute('aria-hidden','true');group.prepend(glider);
      tabGroups.add(group);
      group.addEventListener('click',()=>syncTabs());
      new ResizeObserver(syncTabs).observe(group);
    });
    document.querySelectorAll('dialog').forEach(dialog=>{
      dialog.setAttribute('data-lenis-prevent','');
      dialog.addEventListener('cancel',event=>{event.preventDefault();closeDialog(dialog,{immediate:true});});
      dialog.addEventListener('close',()=>{dialogs.get(dialog)?.cancel();dialogs.delete(dialog);delete dialog.dataset.closing;lenis?.start();});
    });
    document.querySelectorAll('.filmstrip').forEach(element=>element.setAttribute('data-lenis-prevent-horizontal',''));
    document.getElementById('motion-toggle').addEventListener('click',()=>{
      enabled=!enabled;preference=enabled?'full':'reduced';
      try{localStorage.setItem('trace-motion',preference);}catch{}
      updateMotion();
    });
    systemMotion.addEventListener('change',()=>{if(!preference){enabled=!systemMotion.matches;updateMotion();}});
    document.addEventListener('visibilitychange',()=>document.body.classList.toggle('page-hidden',document.hidden));
    updateMotion();
  }

  root.dataset.motion=enabled?'full':'reduced';
  return {get enabled(){return enabled;},get scroller(){return lenis;},mount,disclose,resetDisclosure,openDialog,closeDialog,scrollTo,syncTabs,swapImage,clearImage};
})();
