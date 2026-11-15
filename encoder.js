// Web worker: encodes live camera frames to JPEG off the main thread.
// Keeps one blob per ring-buffer slot, so memory stays small even at full HD.
const blobs = new Map();
let chain = Promise.resolve();
let oc = null;
const run = (fn) => { chain = chain.then(fn).catch((e) => console.error(e)); };
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'frame') run(async () => {
    try {
      if (!oc || oc.width !== m.w || oc.height !== m.h) oc = new OffscreenCanvas(m.w, m.h);
      const x = oc.getContext('2d');
      x.imageSmoothingQuality = 'high';
      x.drawImage(m.bmp, 0, 0, m.w, m.h);
      blobs.set(m.slot, await oc.convertToBlob({ type: 'image/jpeg', quality: m.q }));
    } catch (err) { blobs.delete(m.slot); throw err; }
    finally { m.bmp.close(); self.postMessage({ type: 'done' }); }
  });
  else if (m.type === 'drop') run(() => { blobs.delete(m.slot); });
  else if (m.type === 'get') run(() => { self.postMessage({ type: 'blobs', id: m.id, blobs: m.slots.map((s) => blobs.get(s) ?? null) }); });
  else if (m.type === 'clear') run(() => { blobs.clear(); });
};
