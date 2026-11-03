// Swing analysis core — pure functions, no DOM.
// Coordinates: frames hold landmarks in PIXELS (x right, y down) plus optional
// world landmarks (metres, MediaPipe world space).

export const L = {
  nose: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16,
  lHip: 23, rHip: 24, lKn: 25, rKn: 26, lAn: 27, rAn: 28,
};

export const P_LABELS = {
  P1: 'Ansprechen', P3: 'Halber Rückschwung', P4: 'Top',
  P6: 'Halber Abschwung', P7: 'Treffmoment', P10: 'Finish',
};

// Reference corridors (rough tour/coach guide values, see README)
export const CORRIDORS = {
  tempo:        { label: 'Tempo (Rück : Abschwung)', unit: ':1', lo: 2.7, hi: 3.3, fmt: 1 },
  addrTilt:     { label: 'Oberkörperneigung Ansprechen', unit: '°', lo: 30, hi: 45, fmt: 0 },
  kneeFlex:     { label: 'Kniebeugung Ansprechen', unit: '°', lo: 15, hi: 30, fmt: 0 },
  spineLoss:    { label: 'Neigungsverlust bis Treffmoment', unit: '°', lo: -3, hi: 5, fmt: 0, hint: 'Early Extension / Aufrichten' },
  hipThrust:    { label: 'Hüfte Richtung Ball (Treffm.)', unit: 'cm', lo: -6, hi: 3, fmt: 0, hint: 'Early Extension' },
  headDx:       { label: 'Kopf Richtung Ball (Treffm.)', unit: 'cm', lo: -5, hi: 5, fmt: 0 },
  headDy:       { label: 'Kopf nach unten (Treffm.)', unit: 'cm', lo: -5, hi: 5, fmt: 0 },
  trailElbowTop:{ label: 'Hinterer Ellbogen am Top', unit: '°', lo: 75, hi: 100, fmt: 0 },
  shoulderTurn: { label: 'Schulterdrehung Top (Schätzung)', unit: '°', lo: 85, hi: 110, fmt: 0, est: true },
  hipTurn:      { label: 'Hüftdrehung Top (Schätzung)', unit: '°', lo: 35, hi: 55, fmt: 0, est: true },
  xFactor:      { label: 'X-Factor Top (Schätzung)', unit: '°', lo: 30, hi: 55, fmt: 0, est: true },
};

const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const deg = (r) => (r * 180) / Math.PI;

export function angle3(a, b, c) {
  // angle at b
  const v1 = { x: a.x - b.x, y: a.y - b.y }, v2 = { x: c.x - b.x, y: c.y - b.y };
  const d = (v1.x * v2.x + v1.y * v2.y) / (Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) || 1);
  return deg(Math.acos(Math.max(-1, Math.min(1, d))));
}

export function hands(lm) { return mid(lm[L.lWr], lm[L.rWr]); }
export function hipMid(lm) { return mid(lm[L.lHip], lm[L.rHip]); }
export function shMid(lm) { return mid(lm[L.lSh], lm[L.rSh]); }
export function torso(lm) { return dist(hipMid(lm), shMid(lm)) || 1; }

// Smooth a series of points with a centred moving average, skipping nulls.
function smoothPts(pts, k = 2) {
  return pts.map((_, i) => {
    let sx = 0, sy = 0, n = 0;
    for (let j = i - k; j <= i + k; j++) {
      if (j >= 0 && j < pts.length && pts[j]) { sx += pts[j].x; sy += pts[j].y; n++; }
    }
    return n ? { x: sx / n, y: sy / n } : null;
  });
}

// Fill gaps (frames without pose) by carrying neighbours, so indices stay aligned.
function fillGaps(arr) {
  const out = arr.slice();
  let last = null;
  for (let i = 0; i < out.length; i++) { if (out[i]) last = out[i]; else if (last) out[i] = last; }
  last = null;
  for (let i = out.length - 1; i >= 0; i--) { if (out[i]) last = out[i]; else if (last) out[i] = last; }
  return out;
}

/**
 * Detect swing events in a sequence of frames that contains (at most) one swing.
 * frames: [{ t: seconds, lm: [{x,y,v}] | null, wlm?: [...] }]
 * Returns { ok, reason?, idx: {P1..P10}, dir }
 */
export function detectEvents(frames) {
  const n = frames.length;
  if (n < 12) return { ok: false, reason: 'Zu wenige Bilder' };
  const valid = frames.filter((f) => f.lm).length;
  if (valid < n * 0.6) return { ok: false, reason: 'Körper zu oft nicht erkannt' };

  const H = smoothPts(fillGaps(frames.map((f) => (f.lm ? hands(f.lm) : null))), 1);
  const lmF = fillGaps(frames.map((f) => f.lm));
  const scale = median(lmF.map(torso));

  // hand speed (scale-normalised per frame index; time-agnostic so slow-mo works)
  const sp = H.map((p, i) => (i === 0 ? 0 : dist(p, H[i - 1]) / scale));
  const spS = sp.map((_, i) => avg(sp.slice(Math.max(0, i - 1), i + 2)));

  // impact candidate = fastest hand movement
  let imp0 = argmax(spS, 1, n - 1);
  // top = highest hands (min y) before impact
  let top = argmin(H.map((p) => p.y), 0, imp0);
  if (top <= 1) return { ok: false, reason: 'Kein Rückschwung erkannt' };
  // address height = lowest hands (max y) before top
  const addrIdx = argmax(H.map((p) => p.y), 0, top);
  const addrY = H[addrIdx].y;
  const topY = H[top].y;
  const amp = addrY - topY;
  if (amp < 0.8 * scale) return { ok: false, reason: 'Hände kaum angehoben – kein voller Schwung?' };

  // start of backswing: last frame before top where hands are still near address height
  let start = addrIdx;
  for (let i = top; i >= addrIdx; i--) { if (H[i].y > addrY - 0.03 * amp) { start = i; break; } }
  // P1: a few frames before the start (stable address)
  const p1 = Math.max(0, start - 2);

  // impact: lowest hands between top and the point where hands rise high again
  let endSearch = n - 1;
  for (let i = top + 1; i < n; i++) {
    if (i > imp0 && H[i].y < topY + 0.3 * amp) { endSearch = i; break; }
  }
  const imp = argmax(H.map((p) => p.y), top + 1, endSearch);
  if (imp <= top) return { ok: false, reason: 'Kein Treffmoment erkannt' };
  if (H[imp].y < addrY - 0.45 * amp) return { ok: false, reason: 'Hände kamen nicht zurück zum Ball' };

  // finish: highest hands after impact
  const fin = imp + 1 < n ? argmin(H.map((p) => p.y), imp + 1, n) : imp;

  const firstWhere = (a, b, pred) => { for (let i = a; i <= b; i++) if (pred(i)) return i; return Math.round((a + b) / 2); };
  const p3 = firstWhere(start, top, (i) => H[i].y <= addrY - 0.5 * amp);
  const p6 = firstWhere(top, imp, (i) => H[i].y >= topY + 0.5 * amp);

  // facing direction: nose ahead of hips toward the ball at address
  const a = lmF[p1];
  const dir = Math.sign(a[L.nose].x - hipMid(a).x) || 1;

  return { ok: true, idx: { P1: p1, P3: p3, P4: top, P6: p6, P7: imp, P10: fin }, start, dir, scale };
}

function shoulderLineAngle(w, iA, iB) {
  // rotation angle in horizontal (x-z) plane of world landmarks
  return deg(Math.atan2(w[iB].z - w[iA].z, w[iB].x - w[iA].x));
}
const wrap = (d) => { while (d > 180) d -= 360; while (d < -180) d += 360; return d; };

/**
 * Compute metrics. heightCm = golfer body height for cm estimates.
 */
export function computeMetrics(frames, ev, heightCm = 180) {
  const lmF = fillGaps(frames.map((f) => f.lm));
  const wF = fillGaps(frames.map((f) => f.wlm || null));
  const { idx, dir } = ev;
  const A = lmF[idx.P1], T = lmF[idx.P4], I = lmF[idx.P7];
  // trail side is the one nearer the camera in down-the-line: right side for a right-hander (facing right)
  const trail = dir > 0 ? { sh: L.rSh, el: L.rEl, wr: L.rWr, hip: L.rHip, kn: L.rKn, an: L.rAn }
                        : { sh: L.lSh, el: L.lEl, wr: L.lWr, hip: L.lHip, kn: L.lKn, an: L.lAn };

  // pixel -> cm via leg length (hip->knee->ankle ≈ 0.49 × body height)
  const legPx = dist(A[trail.hip], A[trail.kn]) + dist(A[trail.kn], A[trail.an]);
  const cmPerPx = (0.49 * heightCm) / (legPx || 1);

  const tilt = (lm) => {
    const h = hipMid(lm), s = shMid(lm);
    return deg(Math.atan2(dir * (s.x - h.x), h.y - s.y));
  };

  const m = {};
  const dtBack = frames[idx.P4].t - frames[ev.start].t;
  const dtDown = frames[idx.P7].t - frames[idx.P4].t;
  m.tempo = dtDown > 0 ? dtBack / dtDown : null;
  m.backswingMs = Math.round(dtBack * 1000);
  m.downswingMs = Math.round(dtDown * 1000);
  m.addrTilt = tilt(A);
  m.kneeFlex = 180 - angle3(A[trail.hip], A[trail.kn], A[trail.an]);
  m.spineLoss = tilt(A) - tilt(I);
  m.hipThrust = dir * (hipMid(I).x - hipMid(A).x) * cmPerPx;
  m.headDx = dir * (I[L.nose].x - A[L.nose].x) * cmPerPx;
  m.headDy = (I[L.nose].y - A[L.nose].y) * cmPerPx;
  m.trailElbowTop = angle3(T[trail.sh], T[trail.el], T[trail.wr]);

  const wA = wF[idx.P1], wT = wF[idx.P4];
  if (wA && wT) {
    m.shoulderTurn = Math.abs(wrap(shoulderLineAngle(wT, L.lSh, L.rSh) - shoulderLineAngle(wA, L.lSh, L.rSh)));
    m.hipTurn = Math.abs(wrap(shoulderLineAngle(wT, L.lHip, L.rHip) - shoulderLineAngle(wA, L.lHip, L.rHip)));
    m.xFactor = m.shoulderTurn - m.hipTurn;
  }
  m.handed = dir > 0 ? 'rechts' : 'links';
  return m;
}

export function rate(key, v) {
  const c = CORRIDORS[key];
  if (!c || v == null || Number.isNaN(v)) return 'na';
  if (v >= c.lo && v <= c.hi) return 'good';
  const span = c.hi - c.lo;
  if (v >= c.lo - span * 0.5 && v <= c.hi + span * 0.5) return 'warn';
  return 'bad';
}

export function analyzeSwing(frames, heightCm) {
  const ev = detectEvents(frames);
  if (!ev.ok) return ev;
  return { ...ev, metrics: computeMetrics(frames, ev, heightCm) };
}

// ---------- live segmentation ----------
/**
 * Feed frames in real time; emits a window of frames when a swing completes.
 * Uses wall-clock seconds (t).
 */
export class LiveSegmenter {
  constructor(onSwing) {
    this.onSwing = onSwing;
    this.state = 'wait';      // wait | address | swing | cooldown
    this.buf = [];            // recent frames (ring, ~6s)
    this.stillSince = null;
    this.addrHandY = null;
    this.swingStart = null;
    this.addrStartT = null;
    this.lastMoveT = null;
    this.cooldownUntil = 0;
  }
  reset() { this.state = 'wait'; this.stillSince = null; }
  push(f) {
    this.buf.push(f);
    while (this.buf.length && f.t - this.buf[0].t > 7) this.buf.shift();
    const prev = this.buf.length > 1 ? this.buf[this.buf.length - 2] : null;
    if (!f.lm) return this.state; // tolerate dropped detections
    const sc = torso(f.lm);
    const h = hands(f.lm);
    let speed = 0;
    if (prev && prev.lm) speed = dist(h, hands(prev.lm)) / sc / Math.max(1e-3, f.t - prev.t);
    this.speed = this.speed == null ? speed : this.speed * 0.6 + speed * 0.4;
    const handsLow = h.y > shMid(f.lm).y + 0.6 * sc;

    switch (this.state) {
      case 'cooldown':
        if (f.t > this.cooldownUntil) this.state = 'wait';
        break;
      case 'wait':
        if (handsLow && this.speed < 0.7) {
          this.stillSince ??= f.t;
          if (f.t - this.stillSince > 0.4) { this.state = 'address'; this.addrHandY = h.y; this.addrStartT = this.stillSince; }
        } else this.stillSince = null;
        break;
      case 'address':
        if (handsLow && this.speed < 0.7) this.addrHandY = h.y * 0.2 + this.addrHandY * 0.8;
        if (h.y < this.addrHandY - 0.45 * sc) { this.state = 'swing'; this.swingStart = f.t; this.lastMoveT = f.t; this.awaySince = null; }
        else if (!handsLow && this.speed < 0.5) {
          // hands up but not moving (e.g. walked off) → back to waiting
          this.awaySince ??= f.t;
          if (f.t - this.awaySince > 1.0) { this.state = 'wait'; this.stillSince = null; this.awaySince = null; }
        } else this.awaySince = null;
        break;
      case 'swing': {
        if (this.speed > 1.0) this.lastMoveT = f.t;
        const dur = f.t - this.swingStart;
        if ((dur > 1.0 && f.t - this.lastMoveT > 0.35) || dur > 4) {
          const from = this.swingStart - 1.0;
          const win = this.buf.filter((x) => x.t >= from);
          this.state = 'cooldown';
          this.cooldownUntil = f.t + 1.0;
          this.stillSince = null;
          this.onSwing(win);
        }
        break;
      }
    }
    return this.state;
  }
}

// ---------- helpers ----------
function argmax(a, s = 0, e = a.length) { let b = s; for (let i = s; i < e; i++) if (a[i] > a[b]) b = i; return b; }
function argmin(a, s = 0, e = a.length) { let b = s; for (let i = s; i < e; i++) if (a[i] < a[b]) b = i; return b; }
function avg(a) { return a.reduce((x, y) => x + y, 0) / (a.length || 1); }
function median(a) { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(s.length / 2)] || 1; }
