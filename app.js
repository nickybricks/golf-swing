import { analyzeSwing, LiveSegmenter, CORRIDORS, P_LABELS, rate, L } from './analysis.js';

const MP_VER = '1.2.0';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const MODELS = {
  lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
};
const CONN = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32],[27,29],[28,30]];
const P_KEYS = ['P1','P3','P4','P6','P7','P10'];
const THUMB_LONG = 480;

const $ = (id) => document.getElementById(id);
const state = { swings: [], ref: null, landmarker: null, landmarkerModel: null, lastTs: 0, current: null };

// ---------- settings ----------
const S = {
  get(k, d) { try { return localStorage.getItem('sl_' + k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem('sl_' + k, v); } catch {} },
};
const settingFields = { setHeight: ['height', '180'], setCam: ['cam', 'main'], setModel: ['model', 'full'], setKey: ['key', ''], setClaude: ['claude', 'claude-sonnet-5-5'] };
for (const [id, [k, d]] of Object.entries(settingFields)) {
  $(id).value = S.get(k, d);
  $(id).addEventListener('change', () => S.set(k, $(id).value.trim()));
}
const heightCm = () => Number($('setHeight').value) || 180;

function show(id) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('on', s.id === id));
  window.scrollTo(0, 0);
}

// ---------- MediaPipe ----------
async function getLandmarker(msgEl) {
  const want = $('setModel').value;
  if (state.landmarker && state.landmarkerModel === want) return state.landmarker;
  if (msgEl) msgEl.textContent = 'Lade Pose-Erkennung …';
  const { FilesetResolver, PoseLandmarker } = await import(`${MP_BASE}/vision_bundle.mjs`);
  const vision = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODELS[want], delegate },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  let lm;
  try { lm = await PoseLandmarker.createFromOptions(vision, opts('GPU')); }
  catch (e) { console.warn('GPU failed, CPU fallback', e); lm = await PoseLandmarker.createFromOptions(vision, opts('CPU')); }
  state.landmarker?.close?.();
  state.landmarker = lm; state.landmarkerModel = want;
  return lm;
}

function detect(src, w, h) {
  let ts = performance.now();
  if (ts <= state.lastTs) ts = state.lastTs + 1;
  state.lastTs = ts;
  const r = state.landmarker.detectForVideo(src, ts);
  if (!r.landmarks || !r.landmarks.length) return { lm: null, norm: null, wlm: null };
  const norm = r.landmarks[0];
  const lm = norm.map((p) => ({ x: p.x * w, y: p.y * h, v: p.visibility ?? 1 }));
  const wlm = r.worldLandmarks?.[0]?.map((p) => ({ x: p.x, y: p.y, z: p.z })) || null;
  return { lm, norm: norm.map((p) => ({ x: p.x, y: p.y, v: p.visibility ?? 1 })), wlm };
}

// ---------- drawing ----------
function drawPose(ctx, lm, sx, sy, ox = 0, oy = 0, color = '#3ddc84', lw = 3) {
  if (!lm) return;
  ctx.lineWidth = lw; ctx.strokeStyle = color; ctx.lineCap = 'round';
  for (const [a, b] of CONN) {
    if (!lm[a] || !lm[b]) continue;
    ctx.globalAlpha = Math.min(lm[a].v ?? 1, lm[b].v ?? 1) < 0.3 ? 0.35 : 1;
    ctx.beginPath(); ctx.moveTo(ox + lm[a].x * sx, oy + lm[a].y * sy); ctx.lineTo(ox + lm[b].x * sx, oy + lm[b].y * sy); ctx.stroke();
  }
  ctx.globalAlpha = 1; ctx.fillStyle = '#fff';
  for (const i of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    if (!lm[i]) continue;
    ctx.beginPath(); ctx.arc(ox + lm[i].x * sx, oy + lm[i].y * sy, lw * 1.1, 0, 7); ctx.fill();
  }
}
function drawSpine(ctx, lm, sx, sy, color = '#f5b941') {
  if (!lm) return;
  const h = { x: (lm[23].x + lm[24].x) / 2, y: (lm[23].y + lm[24].y) / 2 };
  const s = { x: (lm[11].x + lm[12].x) / 2, y: (lm[11].y + lm[12].y) / 2 };
  const dx = s.x - h.x, dy = s.y - h.y;
  ctx.setLineDash([6, 6]); ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo((h.x - dx * 0.3) * sx, (h.y - dy * 0.3) * sy); ctx.lineTo((s.x + dx * 0.6) * sx, (s.y + dy * 0.6) * sy); ctx.stroke();
  ctx.setLineDash([]);
}

// ---------- framing check ----------
function framing(norm) {
  if (!norm) return { ok: false, msg: 'Keine Person erkannt', cls: 'bad' };
  const vis = (i) => norm[i].v > 0.5;
  const pair = (a, b) => vis(a) || vis(b);
  const inY = (i) => norm[i].y > 0.01 && norm[i].y < 0.985;
  if (!pair(27, 28) || !(inY(27) || inY(28))) return { ok: false, msg: 'Füße nicht im Bild – Kamera weiter weg / tiefer', cls: 'warn' };
  if (!vis(0)) return { ok: false, msg: 'Kopf nicht sichtbar', cls: 'warn' };
  if (norm[0].y < 0.1) return { ok: false, msg: 'Mehr Platz über dem Kopf (Schläger!)', cls: 'warn' };
  if (!pair(23, 24) || !pair(11, 12)) return { ok: false, msg: 'Oberkörper/Hüfte nicht klar erkennbar', cls: 'warn' };
  if (!pair(15, 16)) return { ok: false, msg: 'Hände nicht sichtbar', cls: 'warn' };
  const xs = [0, 11, 12, 23, 24, 27, 28].map((i) => norm[i].x);
  if (Math.min(...xs) < 0.03 || Math.max(...xs) > 0.97) return { ok: false, msg: 'Zu nah am Bildrand', cls: 'warn' };
  return { ok: true };
}

// ---------- audio / wake lock ----------
let actx = null, wake = null;
function beep(f = 880, d = 0.12) {
  try {
    actx ??= new (window.AudioContext || window.webkitAudioContext)();
    const o = actx.createOscillator(), g = actx.createGain();
    o.frequency.value = f; g.gain.value = 0.15; o.connect(g); g.connect(actx.destination);
    o.start(); o.stop(actx.currentTime + d);
  } catch {}
}
async function keepAwake() { try { wake = await navigator.wakeLock?.request('screen'); } catch {} }

// ---------- live session ----------
const sess = { stream: null, running: false, pool: [], poolIdx: 0, freeze: false, seg: null, fpsN: 0, fpsT: 0, readySince: null, w: 0, h: 0 };

function toast(t) { const el = $('toast'); el.textContent = t; el.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove('show'), 1600); }
function setStatus(t, cls = '') { const el = $('status'); el.textContent = t; el.className = cls; }
function updateCount() {
  const n = state.swings.length;
  $('swCount').textContent = `${n} ${n === 1 ? 'Schwung' : 'Schwünge'}`;
  $('btnResults').style.display = n || state.ref ? '' : 'none';
}

async function openCamera() {
  const base = { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } };
  let stream = await navigator.mediaDevices.getUserMedia({ video: base, audio: false });
  if ($('setCam').value === 'ultra') {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    const uw = devs.find((d) => /ultra/i.test(d.label));
    if (uw) {
      stream.getTracks().forEach((t) => t.stop());
      stream = await navigator.mediaDevices.getUserMedia({ video: { ...base, facingMode: undefined, deviceId: { exact: uw.deviceId } }, audio: false });
    } else toast('Kein Ultraweitwinkel gefunden');
  }
  return stream;
}

async function startSession() {
  show('session');
  $('loading').style.display = 'flex';
  beep(660, 0.01); // unlock audio inside the tap
  keepAwake();
  try {
    const [stream] = await Promise.all([openCamera(), getLandmarker($('loadMsg'))]);
    sess.stream = stream;
    const v = $('video');
    v.srcObject = stream;
    await v.play();
    await new Promise((r) => (v.videoWidth ? r() : v.addEventListener('loadedmetadata', r, { once: true })));
  } catch (e) {
    $('loadMsg').innerHTML = `<span class="err">Fehler: ${e.message || e}</span><br><br>Kamera-Zugriff erlaubt? Läuft die Seite über https?`;
    return;
  }
  $('loading').style.display = 'none';
  const v = $('video');
  sess.w = v.videoWidth; sess.h = v.videoHeight;
  // image ring buffer
  const k = THUMB_LONG / Math.max(sess.w, sess.h);
  const tw = Math.round(sess.w * k), th = Math.round(sess.h * k);
  sess.pool = Array.from({ length: 220 }, () => { const c = document.createElement('canvas'); c.width = tw; c.height = th; return c; });
  sess.poolIdx = 0; sess.freeze = false; sess.readySince = null;
  sess.seg = new LiveSegmenter(onLiveSwing);
  sess.running = true;
  sess.fpsT = performance.now(); sess.fpsN = 0;
  loop();
}

function stopSession() {
  sess.running = false;
  sess.stream?.getTracks().forEach((t) => t.stop());
  sess.stream = null;
  try { wake?.release(); } catch {}
  sess.pool = [];
}

let lastVideoTime = -1;
function loop() {
  if (!sess.running) return;
  const v = $('video');
  const schedule = () => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(loop) : requestAnimationFrame(loop));
  if (v.currentTime === lastVideoTime && !v.requestVideoFrameCallback) return schedule();
  lastVideoTime = v.currentTime;
  const t = performance.now() / 1000;
  let res;
  try { res = detect(v, sess.w, sess.h); } catch (e) { console.error(e); return schedule(); }

  // draw overlay
  const cv = $('overlay');
  const cw = cv.clientWidth * devicePixelRatio, ch = cv.clientHeight * devicePixelRatio;
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cw, ch);
  const s = Math.min(cw / sess.w, ch / sess.h);
  const ox = (cw - sess.w * s) / 2, oy = (ch - sess.h * s) / 2;

  const fr = framing(res.norm);
  const segState = sess.seg.state;
  const col = segState === 'swing' ? '#5aa9ff' : fr.ok ? '#3ddc84' : '#f5b941';
  drawPose(ctx, res.lm, s, s, ox, oy, col, 4 * devicePixelRatio);

  // buffer frame
  let slot = -1;
  if (!sess.freeze && sess.pool.length) {
    slot = sess.poolIdx; sess.poolIdx = (sess.poolIdx + 1) % sess.pool.length;
    const c = sess.pool[slot];
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
  }
  const frame = { t, lm: res.lm, wlm: res.wlm, slot };
  const prevState = sess.seg.state;
  // once the golfer is at address, keep feeding frames even if the framing check flickers
  const tracking = ['address', 'swing'].includes(sess.seg.state);
  const st = fr.ok || tracking ? sess.seg.push(frame) : (sess.seg.state === 'cooldown' ? sess.seg.push(frame) : (sess.seg.reset(), 'wait'));

  // status text
  if (sess.freeze) setStatus('⏳ Werte Schwung aus …', 'ok');
  else if (!fr.ok && st !== 'swing' && st !== 'address') { setStatus(fr.msg, fr.cls); sess.readySince = null; }
  else if (st === 'address') setStatus('🎯 Ansprechen erkannt – los!', 'go');
  else if (st === 'swing') setStatus('🏌️ Schwung läuft …', 'go');
  else if (st === 'cooldown') setStatus('✓ Erfasst', 'ok');
  else setStatus('✓ Bereit – stell dich an den Ball', 'ok');
  if (prevState !== 'address' && st === 'address') beep(1046, 0.06);

  // fps
  sess.fpsN++;
  const now = performance.now();
  if (now - sess.fpsT > 1000) { $('fps').textContent = `${Math.round((sess.fpsN * 1000) / (now - sess.fpsT))} fps Analyse`; sess.fpsN = 0; sess.fpsT = now; }
  schedule();
}

async function onLiveSwing(win) {
  sess.freeze = true;
  try {
    const res = analyzeSwing(win, heightCm());
    if (!res.ok) { toast('Kein Schwung: ' + res.reason); return; }
    const sw = await buildSwing(win, res, (f) => sess.pool[f.slot], sess.w, sess.h, 'live');
    state.swings.push(sw);
    updateCount();
    beep(1318, 0.15);
    toast(`Schwung ${state.swings.length} ✓  Tempo ${fmt(sw.metrics.tempo, 1)}:1`);
  } catch (e) { console.error(e); toast('Fehler bei der Auswertung'); }
  finally { sess.freeze = false; }
}

// Trim frames around the swing, encode images, re-index events.
async function buildSwing(frames, res, getCanvas, w, h, source) {
  const a = Math.max(0, res.idx.P1 - 6), b = Math.min(frames.length - 1, res.idx.P10 + 10);
  const part = frames.slice(a, b + 1);
  const idx = Object.fromEntries(Object.entries(res.idx).map(([k, v]) => [k, v - a]));
  const blobs = [];
  for (const f of part) {
    const c = f.blob ? null : getCanvas(f);
    blobs.push(f.blob || (c ? await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.72)) : null));
  }
  const thumbCanvas = await keyframeCanvas({ frames: part, blobs, w, h }, idx.P4, 270);
  return {
    id: Math.random().toString(36).slice(2, 8), source, created: Date.now(),
    frames: part.map((f) => ({ t: f.t, lm: f.lm, wlm: f.wlm })), blobs, idx, start: res.start - a,
    metrics: res.metrics, w, h, thumb: thumbCanvas.toDataURL('image/jpeg', 0.7),
  };
}

// ---------- video import ----------
async function importVideo(file, asRef) {
  if (!file) return;
  const box = $('importBox'), bar = $('importBar'), msg = $('importMsg');
  box.style.display = ''; bar.style.width = '0%'; msg.textContent = 'Lade Pose-Erkennung …';
  try {
    await getLandmarker(msg);
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.preload = 'auto';
    v.src = URL.createObjectURL(file);
    await new Promise((r, j) => { v.onloadeddata = r; v.onerror = () => j(new Error('Video kann nicht gelesen werden')); });
    try { await v.play(); v.pause(); } catch {}
    const w = v.videoWidth, h = v.videoHeight;
    const dur = Math.min(v.duration || 0, 20);
    const step = dur <= 5 ? 1 / 60 : 1 / 30;
    const k = THUMB_LONG / Math.max(w, h);
    const c = document.createElement('canvas'); c.width = Math.round(w * k); c.height = Math.round(h * k);
    const cx = c.getContext('2d');
    const frames = [];
    for (let t = 0; t <= dur; t += step) {
      await seek(v, t);
      const r = detect(v, w, h);
      cx.drawImage(v, 0, 0, c.width, c.height);
      const blob = await new Promise((rs) => c.toBlob(rs, 'image/jpeg', 0.72));
      frames.push({ t, lm: r.lm, wlm: r.wlm, blob });
      bar.style.width = `${Math.round((t / dur) * 100)}%`;
      msg.textContent = `Analysiere Bild ${frames.length} …`;
    }
    URL.revokeObjectURL(v.src);
    const res = analyzeSwing(frames, asRef ? 180 : heightCm());
    if (!res.ok) throw new Error(res.reason + ' – Clip am besten auf einen Schwung kürzen.');
    const sw = await buildSwing(frames, res, () => null, w, h, asRef ? 'ref' : 'import');
    sw.name = file.name;
    if (asRef) state.ref = sw; else state.swings.push(sw);
    updateCount();
    msg.textContent = asRef ? '⭐️ Referenz geladen.' : '✓ Schwung importiert.';
    bar.style.width = '100%';
    setTimeout(() => { box.style.display = 'none'; openResults(); }, 600);
  } catch (e) {
    console.error(e);
    msg.innerHTML = `<span class="err">${e.message || e}</span>`;
  }
}
function seek(v, t) {
  return new Promise((r) => {
    const done = () => { clearTimeout(to); v.removeEventListener('seeked', done); r(); };
    const to = setTimeout(done, 1500);
    v.addEventListener('seeked', done);
    v.currentTime = t;
  });
}

// ---------- results ----------
const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d).replace('.', ','));
const avgOf = (arr) => { const a = arr.filter((x) => x != null && !Number.isNaN(x)); return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null; };

function metricsTable(rows, cols) {
  // cols: [{title, metrics}]
  let html = `<tr><th>Kennzahl</th>${cols.map((c) => `<th>${c.title}</th>`).join('')}<th>Richtwert</th></tr>`;
  for (const [k, c] of Object.entries(CORRIDORS)) {
    if (rows && !rows.includes(k)) continue;
    html += `<tr><td>${c.label}${c.hint ? `<br><span class="hint">${c.hint}</span>` : ''}</td>`;
    for (const col of cols) {
      const v = col.metrics?.[k];
      html += `<td>${col.dots === false ? '' : `<span class="dot ${rate(k, v)}"></span>`}${fmt(v, c.fmt)}${v == null ? '' : c.unit === ':1' ? ':1' : c.unit}</td>`;
    }
    html += `<td class="hint">${fmt(c.lo, c.fmt)} – ${fmt(c.hi, c.fmt)}${c.unit === ':1' ? ':1' : c.unit}</td></tr>`;
  }
  return html;
}

function sessionAverages() {
  const out = {};
  for (const k of Object.keys(CORRIDORS)) out[k] = avgOf(state.swings.map((s) => s.metrics[k]));
  return out;
}

function openResults() {
  show('results');
  const n = state.swings.length;
  $('resEmpty').style.display = n || state.ref ? 'none' : '';
  $('resBody').style.display = n || state.ref ? '' : 'none';
  const cols = [{ title: `Ø Du (${n})`, metrics: sessionAverages() }];
  if (state.ref) cols.push({ title: '⭐️ Ref.', metrics: state.ref.metrics, dots: false });
  $('sumTable').innerHTML = metricsTable(null, cols);
  const list = $('swList');
  list.innerHTML = '';
  const items = state.ref ? [state.ref, ...state.swings] : state.swings;
  items.forEach((sw) => {
    const i = state.swings.indexOf(sw);
    const b = document.createElement('button');
    b.className = 'sw' + (sw === state.ref ? ' ref' : '');
    b.innerHTML = `<img src="${sw.thumb}"><div><b>${sw === state.ref ? '⭐️ Referenz' : '#' + (i + 1)}</b><br>Tempo ${fmt(sw.metrics.tempo, 1)}:1</div>`;
    b.onclick = () => openDetail(sw);
    list.appendChild(b);
  });
}

// ---------- detail / player ----------
const player = { sw: null, i: 0, playing: false, imgs: [], last: 0, acc: 0 };

async function loadImgs(sw) {
  if (sw._imgs) return sw._imgs;
  sw._imgs = await Promise.all(sw.blobs.map((b) => (b ? createImageBitmap(b) : null)));
  return sw._imgs;
}

async function keyframeCanvas(sw, i, width = 360, label) {
  const imgs = sw._imgs || [];
  const blob = sw.blobs[i];
  const img = imgs[i] || (blob ? await createImageBitmap(blob) : null);
  const ar = sw.h / sw.w;
  const c = document.createElement('canvas'); c.width = width; c.height = Math.round(width * ar);
  const x = c.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
  if (img) x.drawImage(img, 0, 0, c.width, c.height);
  const s = c.width / sw.w;
  drawSpine(x, sw.frames[i].lm, s, s);
  drawPose(x, sw.frames[i].lm, s, s, 0, 0, '#3ddc84', Math.max(2, width / 140));
  if (label) { x.fillStyle = 'rgba(0,0,0,.6)'; x.fillRect(0, 0, c.width, 28); x.fillStyle = '#fff'; x.font = 'bold 16px sans-serif'; x.fillText(label, 8, 20); }
  return c;
}

async function openDetail(sw) {
  show('detail');
  player.sw = sw; player.i = sw.idx.P1; player.playing = false;
  const isRef = sw === state.ref;
  $('detTitle').textContent = isRef ? '⭐️ Referenz' : `Schwung #${state.swings.indexOf(sw) + 1}`;
  $('detDel').onclick = () => {
    if (isRef) state.ref = null; else state.swings.splice(state.swings.indexOf(sw), 1);
    updateCount(); openResults();
  };
  player.imgs = await loadImgs(sw);
  if (state.ref && !isRef) await loadImgs(state.ref);
  const seekEl = $('plSeek'); seekEl.max = sw.frames.length - 1; seekEl.value = player.i;
  // chips
  $('pChips').innerHTML = '';
  for (const k of P_KEYS) {
    const b = document.createElement('button'); b.textContent = `${k} ${P_LABELS[k]}`; b.dataset.k = k;
    b.onclick = () => { player.playing = false; $('plPlay').textContent = '▶︎'; player.i = sw.idx[k]; renderPlayer(); };
    $('pChips').appendChild(b);
  }
  renderPlayer();
  // keyframes grid
  const ref = !isRef && state.ref ? state.ref : null;
  $('kfHint').textContent = ref ? 'links du · rechts Referenz' : '';
  const grid = $('kfGrid'); grid.innerHTML = '';
  grid.style.gridTemplateColumns = ref ? 'repeat(2,1fr)' : 'repeat(3,1fr)';
  for (const k of P_KEYS) {
    for (const s of ref ? [sw, ref] : [sw]) {
      const fig = document.createElement('figure');
      fig.appendChild(await keyframeCanvas(s, s.idx[k], 360));
      const cap = document.createElement('figcaption'); cap.textContent = `${k}${s === ref ? ' ⭐️' : ''}`; cap.title = P_LABELS[k];
      fig.appendChild(cap); grid.appendChild(fig);
    }
  }
  const cols = [{ title: isRef ? 'Ref.' : 'Du', metrics: sw.metrics }];
  if (ref) cols.push({ title: '⭐️ Ref.', metrics: ref.metrics, dots: false });
  $('detTable').innerHTML = metricsTable(null, cols) +
    `<tr><td>Rückschwung / Abschwung</td><td colspan="${cols.length + 1}">${sw.metrics.backswingMs} ms / ${sw.metrics.downswingMs} ms${sw.source !== 'live' ? '<br><span class="hint">bei Zeitlupe nur das Verhältnis aussagekräftig</span>' : ''}</td></tr>`;
}

function renderPlayer() {
  const sw = player.sw; if (!sw) return;
  const cv = $('player');
  const W = Math.min(900, cv.clientWidth * devicePixelRatio || 600);
  const Hh = Math.round(W * sw.h / sw.w);
  if (cv.width !== W || cv.height !== Hh) { cv.width = W; cv.height = Hh; }
  const x = cv.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, W, Hh);
  const img = player.imgs[player.i];
  if (img) x.drawImage(img, 0, 0, W, Hh);
  const s = W / sw.w;
  // address reference lines (posture check)
  const A = sw.frames[sw.idx.P1].lm;
  if (A) {
    x.globalAlpha = 0.5; drawSpine(x, A, s, s, '#ffffff'); x.globalAlpha = 1;
    const hipX = Math.min(A[23].x, A[24].x) * s; // rough butt line helper
    void hipX;
  }
  drawSpine(x, sw.frames[player.i].lm, s, s);
  drawPose(x, sw.frames[player.i].lm, s, s, 0, 0, '#3ddc84', Math.max(3, W / 160));
  $('plSeek').value = player.i;
  const cur = P_KEYS.find((k) => sw.idx[k] === player.i);
  document.querySelectorAll('#pChips button').forEach((b) => b.classList.toggle('on', b.dataset.k === cur));
}

function playTick(ts) {
  if (!player.playing || !player.sw) return;
  const sw = player.sw;
  const dt = player.last ? (ts - player.last) / 1000 : 0; player.last = ts;
  player.acc += dt * Number($('plSpeed').value);
  while (player.i < sw.frames.length - 1 && sw.frames[player.i + 1].t - sw.frames[player.i].t <= player.acc) {
    player.acc -= sw.frames[player.i + 1].t - sw.frames[player.i].t; player.i++;
  }
  if (player.i >= sw.frames.length - 1) { player.playing = false; $('plPlay').textContent = '▶︎'; }
  renderPlayer();
  if (player.playing) requestAnimationFrame(playTick);
}
$('plPlay').onclick = () => {
  if (!player.sw) return;
  player.playing = !player.playing;
  $('plPlay').textContent = player.playing ? '❚❚' : '▶︎';
  if (player.playing) {
    if (player.i >= player.sw.frames.length - 1) player.i = 0;
    player.last = 0; player.acc = 0; requestAnimationFrame(playTick);
  }
};
$('plSeek').oninput = (e) => { player.playing = false; $('plPlay').textContent = '▶︎'; player.i = Number(e.target.value); renderPlayer(); };

// ---------- AI coach ----------
async function contactSheet(sw, title) {
  const w = 240; const cells = [];
  await loadImgs(sw);
  for (const k of P_KEYS) cells.push(await keyframeCanvas(sw, sw.idx[k], w, `${k} ${P_LABELS[k]}`));
  const c = document.createElement('canvas'); c.width = w * cells.length; c.height = cells[0].height + 30;
  const x = c.getContext('2d'); x.fillStyle = '#111'; x.fillRect(0, 0, c.width, c.height);
  x.fillStyle = '#fff'; x.font = 'bold 18px sans-serif'; x.fillText(title, 8, 21);
  cells.forEach((cc, i) => x.drawImage(cc, i * w, 30));
  return c.toDataURL('image/jpeg', 0.8).split(',')[1];
}

function bestRepresentative() {
  // swing whose tempo is closest to the session median (a "typical" swing)
  const sw = state.swings.filter((s) => s.metrics.tempo);
  if (!sw.length) return state.swings.at(-1);
  const med = sw.map((s) => s.metrics.tempo).sort((a, b) => a - b)[Math.floor(sw.length / 2)];
  return sw.reduce((a, b) => (Math.abs(b.metrics.tempo - med) < Math.abs(a.metrics.tempo - med) ? b : a));
}

const round = (m) => Object.fromEntries(Object.entries(m).map(([k, v]) => [k, typeof v === 'number' ? Math.round(v * 10) / 10 : v]));

async function runAI() {
  const key = $('setKey').value.trim();
  const out = $('aiOut'); out.style.display = '';
  if (!key) { out.innerHTML = 'Bitte zuerst in den Einstellungen (Startseite) einen Claude API-Key eintragen.'; return; }
  if (!state.swings.length) { out.textContent = 'Noch keine eigenen Schwünge.'; return; }
  out.innerHTML = '<div class="row"><div class="spin" style="flex:none"></div><div>Claude schaut sich deine Schwünge an …</div></div>';
  $('btnAI').disabled = true;
  try {
    const rep = bestRepresentative();
    const content = [];
    content.push({ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: await contactSheet(rep, `Du – typischer Schwung #${state.swings.indexOf(rep) + 1}`) } });
    if (state.ref) content.push({ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: await contactSheet(state.ref, 'Referenz (Pro)') } });
    const data = {
      perspektive: 'Down-the-Line (Kamera hinter dem Spieler, Blick zum Ziel), Handy-Kamera, 2D-Pose-Erkennung (MediaPipe)',
      haendigkeit: rep.metrics.handed, koerpergroesse_cm: heightCm(),
      kennzahlen_definitionen: Object.fromEntries(Object.entries(CORRIDORS).map(([k, c]) => [k, `${c.label} [${c.unit}], Richtwert ${c.lo}…${c.hi}${c.est ? ' (3D-Schätzung aus 1 Kamera, unsicher)' : ''}`])),
      schwuenge: state.swings.map((s, i) => ({ nr: i + 1, quelle: s.source, ...round(s.metrics) })),
      durchschnitt: round(sessionAverages()),
      referenz: state.ref ? round(state.ref.metrics) : null,
    };
    content.push({ type: 'text', text:
`Hier sind die Daten meiner Golf-Session. Bild 1 zeigt die Schlüsselpositionen P1–P10 meines typischen Schwungs mit erkanntem Skelett (gelb gestrichelt = Wirbelsäulenachse).${state.ref ? ' Bild 2 zeigt dieselben Positionen eines Referenz-Schwungs (Pro).' : ''}

${JSON.stringify(data, null, 1)}

Bitte analysiere wie ein erfahrener Golflehrer auf Deutsch, knapp und konkret:
### Kurzfazit (2–3 Sätze)
### Die 3 wichtigsten Baustellen – jeweils: was du siehst (Bild + Zahl), warum es den Ballflug beeinflusst, ein konkreter Drill
### Was schon gut ist
${state.ref ? '### Größte Unterschiede zur Referenz\n' : ''}### Konstanz über die Session (Streuung der Werte)
Weise darauf hin, wenn ein Wert wahrscheinlich ein Messfehler ist (z. B. Hände verdeckt, Schätzwerte). Keine Einleitung.` });

    const resp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
      body: JSON.stringify({ model: $('setClaude').value.trim() || 'claude-sonnet-5-5', max_tokens: 2000, messages: [{ role: 'user', content }] }),
    });
    const j = await resp.json();
    if (!resp.ok) throw new Error(j.error?.message || resp.status);
    out.innerHTML = md(j.content.filter((c) => c.type === 'text').map((c) => c.text).join('\n'));
  } catch (e) {
    out.innerHTML = `<span class="err">Fehler: ${e.message || e}</span>`;
  } finally { $('btnAI').disabled = false; }
}

function md(t) {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\*(.+?)\*/g, '<i>$1</i>');
  let html = '', inList = false;
  for (const line of t.split('\n')) {
    const l = line.trim();
    if (/^[-*•] /.test(l) || /^\d+\. /.test(l)) { if (!inList) { html += '<ul>'; inList = true; } html += `<li>${inline(l.replace(/^([-*•]|\d+\.) /, ''))}</li>`; continue; }
    if (inList) { html += '</ul>'; inList = false; }
    if (/^#{1,4} /.test(l)) html += `<h3>${inline(l.replace(/^#+ /, ''))}</h3>`;
    else if (l) html += `<p>${inline(l)}</p>`;
  }
  return html + (inList ? '</ul>' : '');
}

// ---------- wiring ----------
$('btnStart').onclick = startSession;
$('btnStop').onclick = () => { stopSession(); show('home'); updateCount(); };
$('btnAnalyze').onclick = () => { stopSession(); openResults(); };
$('btnResults').onclick = openResults;
$('resBack').onclick = () => show('home');
$('resCam').onclick = startSession;
$('detBack').onclick = () => { player.playing = false; openResults(); };
$('btnAI').onclick = runAI;
$('btnImport').onclick = () => $('fileImport').click();
$('btnRef').onclick = () => $('fileRef').click();
$('fileImport').onchange = (e) => { importVideo(e.target.files[0], false); e.target.value = ''; };
$('fileRef').onchange = (e) => { importVideo(e.target.files[0], true); e.target.value = ''; };
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && sess.running) keepAwake(); });
updateCount();

// expose for testing
window.__swing = { state, sess, importVideo, openResults, runAI, buildSwing, analyzeSwing, openDetail, updateCount };
void L;
