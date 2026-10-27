/* Keep one accessible image/player while following Furo's effective theme. */
(() => {
  'use strict';
  const preference = window.matchMedia('(prefers-color-scheme: dark)');
  const pending = new WeakMap();
  let serial = 0;
  const absolute = value => new URL(value, document.baseURI).href;

  function darkMode() {
    const mode = document.body.dataset.theme || 'auto';
    return mode === 'dark' || (mode !== 'light' && preference.matches);
  }

  function pictureTheme(picture, dark) {
    const source = picture.querySelector('source[data-theme-dark]');
    source.media = dark ? 'all' : 'not all';
    const link = picture.closest('a.image-reference');
    if (link) {
      const light = absolute(picture.dataset.themeLightSrc);
      const darkURL = absolute(picture.dataset.themeDarkSrc);
      if (link.href === light || link.href === darkURL) link.href = dark ? darkURL : light;
    }
  }

  function videoTheme(video, dark) {
    const wanted = dark ? video.dataset.themeDarkSrc : video.dataset.themeLightSrc;
    const poster = dark ? video.dataset.themeDarkPoster : video.dataset.themeLightPoster;
    if (poster) video.poster = poster;
    const source = video.querySelector('source');
    const current = source ? source.getAttribute('src') : video.getAttribute('src');
    if (!wanted || (current && absolute(current) === absolute(wanted))) return;

    const old = pending.get(video);
    // Preserve the original intent during rapid toggles while a replacement loads.
    const intent = old && old.loading ? old.intent : {
      time: Number.isFinite(video.currentTime) ? video.currentTime : 0,
      playing: !video.paused && !video.ended,
      rate: video.playbackRate,
      muted: video.muted,
      volume: video.volume,
      preload: video.preload,
    };
    const state = { id: ++serial, loading: true, intent };
    pending.set(video, state);
    video.pause();
    if (source) source.src = wanted;
    else video.src = wanted;
    video.playbackRate = intent.rate;
    video.muted = intent.muted;
    video.volume = intent.volume;

    const restore = () => {
      if (pending.get(video) !== state || absolute(video.currentSrc) !== absolute(wanted)) return;
      state.loading = false;
      if (intent.time > 0) {
        const end = Number.isFinite(video.duration) ? Math.max(0, video.duration - .001) : intent.time;
        try { video.currentTime = Math.min(intent.time, end); } catch (_) { /* codec not seekable */ }
      }
      video.playbackRate = intent.rate;
      video.muted = intent.muted;
      video.volume = intent.volume;
      video.preload = intent.preload;
      if (intent.playing) video.play().catch(() => { /* preserve browser autoplay policy */ });
    };
    if (intent.playing || intent.time > 0) {
      // A paused preload=none replacement would otherwise never deliver metadata.
      video.preload = 'metadata';
      video.addEventListener('loadedmetadata', restore, { once: true });
    } else {
      state.loading = false;  // an untouched poster needs no video download or seek
    }
    video.load();
  }

  function sync() {
    const dark = darkMode();
    document.querySelectorAll('a[data-theme-light-href]').forEach(link => {
      link.href = dark ? link.dataset.themeDarkHref : link.dataset.themeLightHref;
    });
    document.querySelectorAll('picture.themed-media-picture').forEach(picture => pictureTheme(picture, dark));
    document.querySelectorAll('video.themed-media-video').forEach(video => videoTheme(video, dark));
  }

  function start() {
    new MutationObserver(sync).observe(document.body, { attributes: true, attributeFilter: ['data-theme'] });
    preference.addEventListener('change', sync);
    sync();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
  else start();
})();
