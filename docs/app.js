import { predict } from "./infer.js";

// ---------- elements ----------
const $ = (id) => document.getElementById(id);
const pad = $("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
const scoreEl = $("score"), timerEl = $("timer"), bestEl = $("best");
const qaEl = $("qa"), qbEl = $("qb"), opEl = $("op"), qresEl = $("qres");
const verdictEl = $("verdict"), timerWrap = $("timerWrap");
const overlay = $("overlay"), overlayMsg = $("overlayMsg");

// ---------- state ----------
const IDLE_MS = 470;        // wait after last stroke before recognizing
const ROUND_SEC = 60;
const BEST_KEY = "onikeisan.best";

let mode = "oni";           // "oni" (60s) | "practice"
let playing = false;
let score = 0;
let timeLeft = ROUND_SEC;
let tick = null;
let expected = 0;
let hasInk = false;
let recogTimer = null;
let locked = false;         // ignore input while showing verdict

bestEl.textContent = (+localStorage.getItem(BEST_KEY) || 0);

// ---------- canvas sizing ----------
let dpr = 1, strokeW = 14;
function resize() {
  const rect = pad.getBoundingClientRect();
  dpr = Math.min(window.devicePixelRatio || 1, 3);
  pad.width = Math.round(rect.width * dpr);
  pad.height = Math.round(rect.height * dpr);
  strokeW = Math.max(10, Math.min(rect.width, rect.height) * 0.05) * dpr;
  clearPad();
}
function clearPad() {
  ctx.clearRect(0, 0, pad.width, pad.height);
  hasInk = false;
}
window.addEventListener("resize", resize);
window.addEventListener("orientationchange", () => setTimeout(resize, 200));

// ---------- drawing ----------
let drawing = false, lastX = 0, lastY = 0;
function posOf(e) {
  const rect = pad.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (pad.width / rect.width),
    y: (e.clientY - rect.top) * (pad.height / rect.height),
  };
}
function strokeBegin(e) {
  if (locked || !playing) return;
  e.preventDefault();
  if (recogTimer) { clearTimeout(recogTimer); recogTimer = null; }
  drawing = true;
  const p = posOf(e);
  lastX = p.x; lastY = p.y;
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.strokeStyle = "#f4f6f8"; ctx.lineWidth = strokeW;
  // dot for single taps
  ctx.beginPath();
  ctx.arc(lastX, lastY, strokeW / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#f4f6f8"; ctx.fill();
  hasInk = true;
}
function strokeMove(e) {
  if (!drawing) return;
  e.preventDefault();
  const p = posOf(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x; lastY = p.y;
}
function strokeEnd(e) {
  if (!drawing) return;
  e && e.preventDefault();
  drawing = false;
  if (hasInk) recogTimer = setTimeout(recognize, IDLE_MS);
}
pad.addEventListener("pointerdown", strokeBegin);
pad.addEventListener("pointermove", strokeMove);
pad.addEventListener("pointerup", strokeEnd);
pad.addEventListener("pointercancel", strokeEnd);
pad.addEventListener("pointerleave", strokeEnd);

// ---------- preprocessing: canvas ink -> 28x28 MNIST-like Float32 ----------
const tmp = document.createElement("canvas");
tmp.width = 28; tmp.height = 28;
const tctx = tmp.getContext("2d", { willReadFrequently: true });

function getInkGrid() {
  const W = pad.width, H = pad.height;
  const data = ctx.getImageData(0, 0, W, H).data;
  // bounding box of ink (alpha channel)
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (data[(y * W + x) * 4 + 3] > 24) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // empty
  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const dw = Math.max(1, Math.round(bw * scale));
  const dh = Math.max(1, Math.round(bh * scale));
  const ox = (28 - dw) / 2, oy = (28 - dh) / 2;

  tctx.clearRect(0, 0, 28, 28);
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(pad, minX, minY, bw, bh, ox, oy, dw, dh);

  const td = tctx.getImageData(0, 0, 28, 28).data;
  const raw = new Float32Array(784);
  let sum = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {
    const v = td[i * 4 + 3] / 255; // alpha = ink
    raw[i] = v;
    sum += v; cx += (i % 28) * v; cy += ((i / 28) | 0) * v;
  }
  if (sum === 0) return null;
  // center on center-of-mass (MNIST style)
  const shiftX = Math.round(14 - cx / sum);
  const shiftY = Math.round(14 - cy / sum);
  if (shiftX === 0 && shiftY === 0) return raw;
  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    const sy = y - shiftY;
    if (sy < 0 || sy >= 28) continue;
    for (let x = 0; x < 28; x++) {
      const sx = x - shiftX;
      if (sx < 0 || sx >= 28) continue;
      out[y * 28 + x] = raw[sy * 28 + sx];
    }
  }
  return out;
}

// ---------- recognition ----------
function recognize() {
  recogTimer = null;
  if (locked || !playing) return;
  const grid = getInkGrid();
  if (!grid) return;
  const { digit, probs } = predict(grid);
  // We know the expected answer, so be lenient: accept if it's the top
  // prediction OR a confident runner-up. Reduces false rejects on messy strokes.
  const correct = digit === expected || probs[expected] >= 0.34;
  showVerdict(correct ? expected : digit, correct);
}

function showVerdict(num, ok) {
  locked = true;
  verdictEl.textContent = num;
  verdictEl.className = ""; void verdictEl.offsetWidth; // restart animation
  verdictEl.classList.add("show", ok ? "ok" : "bad");
  vibrate(ok ? 30 : [0, 25, 45, 25]);
  setTimeout(() => {
    locked = false;
    clearPad();
    if (ok) { score++; updateHud(); nextProblem(); }
  }, ok ? 360 : 520);
}

function vibrate(p) { if (navigator.vibrate) try { navigator.vibrate(p); } catch (_) {} }

// ---------- problems ----------
function rnd(n) { return Math.floor(Math.random() * n); }
function nextProblem() {
  let a, b, op;
  if (Math.random() < 0.5) {           // addition, single-digit, sum 0..9
    a = rnd(10); b = rnd(10 - a); op = "+"; expected = a + b;
  } else {                              // subtraction, a>=b, both single-digit
    a = rnd(10); b = rnd(a + 1); op = "−"; expected = a - b;
  }
  qaEl.textContent = a; qbEl.textContent = b; opEl.textContent = op;
  qresEl.textContent = "?";
}

// ---------- game flow ----------
function updateHud() {
  scoreEl.textContent = score;
  timerEl.textContent = timeLeft;
}
function startRound() {
  overlay.classList.add("hidden");
  score = 0; timeLeft = ROUND_SEC; playing = true; locked = false;
  timerWrap.style.visibility = mode === "oni" ? "visible" : "hidden";
  timerWrap.classList.remove("warn");
  updateHud();
  clearPad();
  nextProblem();
  if (tick) clearInterval(tick);
  if (mode === "oni") {
    tick = setInterval(() => {
      timeLeft--;
      timerEl.textContent = timeLeft;
      if (timeLeft <= 10) timerWrap.classList.add("warn");
      if (timeLeft <= 0) endRound();
    }, 1000);
  }
}
function endRound() {
  playing = false;
  clearInterval(tick); tick = null;
  if (recogTimer) { clearTimeout(recogTimer); recogTimer = null; }
  const best = +localStorage.getItem(BEST_KEY) || 0;
  if (score > best) { localStorage.setItem(BEST_KEY, score); bestEl.textContent = score; }
  overlayMsg.textContent = `スコア ${score}\n${score > best ? "🎉 ベストこうしん！" : `ベスト ${Math.max(best, score)}`}`;
  $("startBtn").textContent = "もういちど";
  overlay.classList.remove("hidden");
}

$("startBtn").addEventListener("click", startRound);
$("clearBtn").addEventListener("click", () => { if (!locked) clearPad(); });
$("modeBtn").addEventListener("click", () => {
  mode = mode === "oni" ? "practice" : "oni";
  $("modeBtn").textContent = mode === "oni" ? "れんしゅう" : "おにモード";
  if (playing) startRound();
});

// ---------- boot ----------
resize();
// warm up the model so the first real recognition is instant
predict(new Float32Array(784));

// offline support
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js").catch(() => {}));
}
