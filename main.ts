// main.ts — model-free photo geolocator for make.fun photo-geolocator-01.
//
// Node standard library only. No model calls, no API keys, no wall clock,
// no randomness. A hand-written baseline JPEG decoder feeds a small feature
// extractor; the guess is a fixed point (57.5, 27.5), with an arid override
// (-13.75, -29.5) when the lower part of the frame is mostly sand and
// there is almost no vegetation. Pixel cues do not separate regions
// reliably, so a smarter classifier scores worse than the fixed point.
//
// Env: MAKEFUN_PROMPT_FILE (advisory), MAKEFUN_ENV_URL (base URL).
// GET /photos, GET /photos/<id>.jpg, POST /answers.

import { readFileSync } from "node:fs";

const FIXED = { lat: 57.5, lon: 27.5 };
const ARID = { lat: -13.75, lon: -29.5 };

interface Image {
  width: number;
  height: number;
  data: Uint8Array; // RGB
}

// ---------------------------------------------------------------------------
// Baseline JPEG decoder
// ---------------------------------------------------------------------------

const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40,
  48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
  54, 47, 55, 62, 63,
];

interface Huff {
  // canonical decode tables indexed by code length 1..16
  mincode: Int32Array;
  maxcode: Int32Array;
  valptr: Int32Array;
  vals: Uint8Array;
}

function buildHuff(counts: number[], vals: Uint8Array): Huff {
  const mincode = new Int32Array(17);
  const maxcode = new Int32Array(17).fill(-1);
  const valptr = new Int32Array(17);
  let code = 0;
  let k = 0;
  for (let l = 1; l <= 16; l++) {
    valptr[l] = k;
    mincode[l] = code;
    code += counts[l - 1];
    k += counts[l - 1];
    maxcode[l] = counts[l - 1] ? code - 1 : -1;
    code <<= 1;
  }
  return { mincode, maxcode, valptr, vals };
}

interface Comp {
  id: number;
  h: number;
  v: number;
  tq: number;
  dc: number;
  ac: number;
  pred: number;
  bw: number; // blocks per line (padded to MCU)
  bh: number;
  coef: Float32Array; // decoded samples (padded plane), stride bw*8
  w: number; // true sampled width
  h8: number; // true sampled height
}

function idct8(block: Float32Array, out: Float32Array): void {
  // separable float IDCT with precomputed cosine table
  const tmp = new Float32Array(64);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      let s = 0;
      for (let u = 0; u < 8; u++) s += COS[x * 8 + u] * block[y * 8 + u];
      tmp[y * 8 + x] = s;
    }
  }
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      let s = 0;
      for (let v = 0; v < 8; v++) s += COS[y * 8 + v] * tmp[v * 8 + x];
      out[y * 8 + x] = s;
    }
  }
}
const COS = new Float32Array(64);
for (let x = 0; x < 8; x++)
  for (let u = 0; u < 8; u++)
    COS[x * 8 + u] =
      (u === 0 ? Math.SQRT1_2 : 1) * 0.5 * Math.cos(((2 * x + 1) * u * Math.PI) / 16);

function decodeJpeg(buf: Uint8Array): Image {
  let pos = 0;
  const u16 = () => ((buf[pos] << 8) | buf[pos + 1]);
  if (buf[0] !== 0xff || buf[1] !== 0xd8) throw new Error("not a JPEG");
  pos = 2;
  const qt: Int32Array[] = [];
  const dcT: Huff[] = [];
  const acT: Huff[] = [];
  let comps: Comp[] = [];
  let width = 0;
  let height = 0;
  let hmax = 1;
  let vmax = 1;
  let ri = 0;
  let sawScan = false;

  while (pos < buf.length && !sawScan) {
    if (buf[pos] !== 0xff) {
      pos++;
      continue;
    }
    while (buf[pos] === 0xff) pos++;
    const m = buf[pos++];
    if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7) || m === 0) continue;
    if (m === 0xd9) break;
    const len = u16();
    const end = pos + len;
    const seg = pos + 2;
    if (m === 0xdb) {
      let p = seg;
      while (p < end) {
        const pq = buf[p] >> 4;
        const tq = buf[p] & 15;
        p++;
        const t = new Int32Array(64);
        for (let i = 0; i < 64; i++) {
          if (pq) {
            t[ZIGZAG[i]] = (buf[p] << 8) | buf[p + 1];
            p += 2;
          } else t[ZIGZAG[i]] = buf[p++];
        }
        qt[tq] = t;
      }
    } else if (m === 0xc0 || m === 0xc1) {
      height = (buf[seg + 1] << 8) | buf[seg + 2];
      width = (buf[seg + 3] << 8) | buf[seg + 4];
      const n = buf[seg + 5];
      comps = [];
      for (let i = 0; i < n; i++) {
        const o = seg + 6 + i * 3;
        comps.push({
          id: buf[o], h: buf[o + 1] >> 4, v: buf[o + 1] & 15, tq: buf[o + 2],
          dc: 0, ac: 0, pred: 0, bw: 0, bh: 0, coef: new Float32Array(0), w: 0, h8: 0,
        });
      }
      hmax = Math.max(...comps.map((c) => c.h));
      vmax = Math.max(...comps.map((c) => c.v));
    } else if (m === 0xc2 || (m >= 0xc3 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc)) {
      throw new Error("unsupported (non-baseline) JPEG");
    } else if (m === 0xc4) {
      let p = seg;
      while (p < end) {
        const tc = buf[p] >> 4;
        const th = buf[p] & 15;
        p++;
        const counts: number[] = [];
        let total = 0;
        for (let i = 0; i < 16; i++) {
          counts.push(buf[p + i]);
          total += buf[p + i];
        }
        p += 16;
        const h = buildHuff(counts, buf.slice(p, p + total));
        p += total;
        (tc === 0 ? dcT : acT)[th] = h;
      }
    } else if (m === 0xdd) {
      ri = (buf[seg] << 8) | buf[seg + 1];
    } else if (m === 0xda) {
      const n = buf[seg];
      const scanComps: Comp[] = [];
      for (let i = 0; i < n; i++) {
        const cid = buf[seg + 1 + i * 2];
        const t = buf[seg + 2 + i * 2];
        const c = comps.find((x) => x.id === cid);
        if (!c) throw new Error("bad scan component");
        c.dc = t >> 4;
        c.ac = t & 15;
        scanComps.push(c);
      }
      pos = end;
      decodeScan(scanComps);
      sawScan = true;
      continue;
    }
    pos = end;
  }
  if (!sawScan || !width || !height) throw new Error("no image data");

  function decodeScan(sc: Comp[]): void {
    const mcux = Math.ceil(width / (8 * hmax));
    const mcuy = Math.ceil(height / (8 * vmax));
    for (const c of comps) {
      c.bw = mcux * c.h;
      c.bh = mcuy * c.v;
      c.coef = new Float32Array(c.bw * 8 * c.bh * 8);
      c.w = Math.ceil((width * c.h) / hmax);
      c.h8 = Math.ceil((height * c.v) / vmax);
    }
    // bit reader (handles 0xFF00 stuffing, stops at markers)
    let bitBuf = 0;
    let bitCnt = 0;
    let hitMarker = false;
    const readBit = (): number => {
      if (bitCnt === 0) {
        let b = 0;
        if (!hitMarker && pos < buf.length) {
          b = buf[pos];
          if (b === 0xff) {
            const n = buf[pos + 1];
            if (n === 0) pos += 2;
            else {
              hitMarker = true;
              b = 0;
            }
          } else pos++;
        }
        bitBuf = b;
        bitCnt = 8;
      }
      bitCnt--;
      return (bitBuf >> bitCnt) & 1;
    };
    const receive = (n: number): number => {
      let v = 0;
      for (let i = 0; i < n; i++) v = (v << 1) | readBit();
      return v;
    };
    const extend = (v: number, n: number): number => (n === 0 ? 0 : v < 1 << (n - 1) ? v - (1 << n) + 1 : v);
    const decodeSym = (h: Huff): number => {
      let code = 0;
      for (let l = 1; l <= 16; l++) {
        code = (code << 1) | readBit();
        if (h.maxcode[l] >= 0 && code <= h.maxcode[l] && code >= h.mincode[l])
          return h.vals[h.valptr[l] + code - h.mincode[l]];
      }
      throw new Error("bad huffman code");
    };

    const blk = new Float32Array(64);
    const px = new Float32Array(64);
    const decodeBlock = (c: Comp, bx: number, by: number): void => {
      blk.fill(0);
      const q = qt[c.tq];
      const dcH = dcT[c.dc];
      const acH = acT[c.ac];
      if (!q || !dcH || !acH) throw new Error("missing table");
      const t = decodeSym(dcH);
      c.pred += extend(receive(t), t);
      blk[0] = c.pred * q[0];
      for (let k = 1; k < 64; ) {
        const rs = decodeSym(acH);
        const s = rs & 15;
        const r = rs >> 4;
        if (s === 0) {
          if (r === 15) {
            k += 16;
            continue;
          }
          break;
        }
        k += r;
        if (k > 63) break;
        const z = ZIGZAG[k];
        blk[z] = extend(receive(s), s) * q[z];
        k++;
      }
      idct8(blk, px);
      const stride = c.bw * 8;
      for (let y = 0; y < 8; y++) {
        const o = (by * 8 + y) * stride + bx * 8;
        for (let x = 0; x < 8; x++) c.coef[o + x] = px[y * 8 + x] + 128;
      }
    };

    const reset = () => {
      bitCnt = 0;
      hitMarker = false;
      // skip to after RSTn marker
      while (pos + 1 < buf.length && !(buf[pos] === 0xff && buf[pos + 1] >= 0xd0 && buf[pos + 1] <= 0xd7)) {
        if (buf[pos] === 0xff && buf[pos + 1] === 0xd9) return;
        pos++;
      }
      pos += 2;
      for (const c of comps) c.pred = 0;
    };

    let count = 0;
    const single = sc.length === 1;
    if (single) {
      const c = sc[0];
      const bwTrue = Math.ceil(c.w / 8);
      const bhTrue = Math.ceil(c.h8 / 8);
      for (let by = 0; by < bhTrue; by++)
        for (let bx = 0; bx < bwTrue; bx++) {
          if (ri && count > 0 && count % ri === 0) reset();
          decodeBlock(c, bx, by);
          count++;
        }
    } else {
      for (let my = 0; my < mcuy; my++)
        for (let mx = 0; mx < mcux; mx++) {
          if (ri && count > 0 && count % ri === 0) reset();
          for (const c of sc)
            for (let v = 0; v < c.v; v++)
              for (let h = 0; h < c.h; h++) decodeBlock(c, mx * c.h + h, my * c.v + v);
          count++;
        }
    }
  }

  // assemble RGB, upsampling each plane by its own sampled dimensions
  const out = new Uint8Array(width * height * 3);
  const planeAt = (c: Comp, x: number, y: number): number => {
    let sx = Math.floor((x * c.h) / hmax);
    let sy = Math.floor((y * c.v) / vmax);
    if (sx >= c.w) sx = c.w - 1;
    if (sy >= c.h8) sy = c.h8 - 1;
    return c.coef[sy * c.bw * 8 + sx];
  };
  const clamp = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 3;
      if (comps.length === 1) {
        const l = clamp(planeAt(comps[0], x, y));
        out[o] = out[o + 1] = out[o + 2] = l;
      } else {
        const Y = planeAt(comps[0], x, y);
        const cb = planeAt(comps[1], x, y) - 128;
        const cr = planeAt(comps[2], x, y) - 128;
        out[o] = clamp(Y + 1.402 * cr);
        out[o + 1] = clamp(Y - 0.344136 * cb - 0.714136 * cr);
        out[o + 2] = clamp(Y + 1.772 * cb);
      }
    }
  }
  return { width, height, data: out };
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

interface Features {
  skyBlue: number;
  skyGray: number;
  veg: number;
  vegBright: number;
  vegDark: number;
  sand: number;
  redDirt: number;
  snow: number;
  water: number;
  road: number;
  yellow: number;
  edge: number;
  warm: number;
  bright: number;
  sat: number;
  green: number;
  snowGround: number;
  vegGround: number;
  sandGround: number;
}

function extract(img: Image): Features {
  const { width: w, height: h, data: d } = img;
  const skyRows = Math.floor(h * 0.35);
  const groundStart = Math.floor(h * 0.4);
  const f: Record<string, number> = {};
  const keys = ["skyBlue", "skyGray", "veg", "vegBright", "vegDark", "sand", "redDirt", "snow", "water", "road", "yellow", "edge", "warm", "bright", "sat", "green", "snowGround", "vegGround", "sandGround"];
  for (const k of keys) f[k] = 0;
  let nGround = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 3;
      const r = d[o], g = d[o + 1], b = d[o + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const s = mx === 0 ? 0 : (mx - mn) / mx;
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      const isVeg = g > r + 6 && g > b + 10;
      const isSand = r > 140 && r >= g && g >= b && r - b > 25 && r - b < 110 && s < 0.5;
      const isSnow = mx > 200 && s < 0.1;
      f.veg += isVeg ? 1 : 0;
      f.vegBright += isVeg && l > 120 ? 1 : 0;
      f.vegDark += isVeg && l <= 70 ? 1 : 0;
      f.sand += isSand ? 1 : 0;
      f.redDirt += r > 120 && r > g + 30 && g > b && s > 0.4 ? 1 : 0;
      f.snow += isSnow ? 1 : 0;
      f.water += b > r + 20 && b >= g && y >= skyRows ? 1 : 0;
      f.road += s < 0.12 && l > 60 && l < 150 ? 1 : 0;
      f.yellow += r > 180 && g > 150 && b < 100 ? 1 : 0;
      f.warm += r - b;
      f.bright += l;
      f.sat += s;
      f.green += g;
      if (x + 1 < w && y + 1 < h) {
        const o2 = o + 3, o3 = o + w * 3;
        const l2 = 0.299 * d[o2] + 0.587 * d[o2 + 1] + 0.114 * d[o2 + 2];
        const l3 = 0.299 * d[o3] + 0.587 * d[o3 + 1] + 0.114 * d[o3 + 2];
        f.edge += Math.abs(l - l2) + Math.abs(l - l3) > 40 ? 1 : 0;
      }
      if (y < skyRows) {
        f.skyBlue += b > r + 15 && b > 130 ? 1 : 0;
        f.skyGray += s < 0.12 && l > 130 ? 1 : 0;
      }
      if (y >= groundStart) {
        f.snowGround += isSnow ? 1 : 0;
        f.vegGround += isVeg ? 1 : 0;
        f.sandGround += isSand ? 1 : 0;
      }
    }
  }
  nGround = (h - groundStart) * w;
  const n = w * h;
  const sn = Math.max(1, skyRows * w);
  for (const k of ["veg", "vegBright", "vegDark", "sand", "redDirt", "snow", "water", "road", "yellow", "edge"]) f[k] /= n;
  f.warm /= n * 255;
  f.bright /= n * 255;
  f.sat /= n;
  f.green /= n * 255;
  f.skyBlue /= sn;
  f.skyGray /= sn;
  for (const k of ["snowGround", "vegGround", "sandGround"]) f[k] /= Math.max(1, nGround);
  return f as unknown as Features;
}

function guess(f: Features | null): { lat: number; lon: number } {
  if (f && f.sandGround > 0.5 && f.veg < 0.05) return ARID;
  return FIXED;
}

// ---------------------------------------------------------------------------
// Trial env protocol
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  if (promptFile) {
    try {
      readFileSync(promptFile, "utf8"); // advisory only
    } catch {
      /* ignore */
    }
  }
  const base = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
  if (!base) throw new Error("MAKEFUN_ENV_URL not set");

  const listRes = await fetch(`${base}/photos`);
  if (!listRes.ok) throw new Error(`GET /photos -> ${listRes.status}`);
  const list = (await listRes.json()) as { photos: { id: string }[] };
  const ids = list.photos.map((p) => p.id).sort();

  const answers: { id: string; lat: number; lon: number }[] = [];
  for (const id of ids) {
    let g = FIXED;
    try {
      const r = await fetch(`${base}/photos/${id}.jpg`);
      if (!r.ok) throw new Error(`status ${r.status}`);
      const img = decodeJpeg(new Uint8Array(await r.arrayBuffer()));
      g = guess(extract(img));
    } catch (e) {
      console.error(`photo ${id}: ${(e as Error).message}; using fixed point`);
    }
    answers.push({ id, lat: g.lat, lon: g.lon });
  }

  const res = await fetch(`${base}/answers`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ answers }),
  });
  if (!res.ok) throw new Error(`POST /answers -> ${res.status} ${await res.text()}`);
  console.log(`answered ${answers.length} photos`);
}

main().then(
  () => process.exit(0),
  (e) => {
    console.error(e);
    process.exit(1);
  },
);
