import { predict } from "./infer.js";

// ====== 鬼計算 = N-back arithmetic (Dr. Kawashima rules) ======
// - Each problem is shown one at a time; you only MEMORIZE the current one.
// - You write the answer of the problem N steps back (1-back, 2-back, ...).
// - 1 set = 22 problems. After each set, accuracy decides the level:
//     >=85% level up, 66-84% stay, <=65% level down (min 1).
// - When 5 minutes pass, the current set finishes and the game ends.
// - Record = highest back-level reached.

const $ = (id) => document.getElementById(id);
const pad = $("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
const levelEl = $("level"), timerEl = $("timer"), bestEl = $("best");
const qaEl = $("qa"), qbEl = $("qb"), opEl = $("op");
const problemEl = $("problem"), instructEl = $("instruct"), padWrap = $("padWrap");
const verdictEl = $("verdict"), timerWrap = $("timerWrap");
const progressFill = $("progressFill");
const toastEl = $("toast");
const overlay = $("overlay"), overlayMsg = $("overlayMsg"), startBtn = $("startBtn");
const nextBtn = $("nextBtn"), clearBtn = $("clearBtn"), passBtn = $("passBtn");

const IDLE_MS = 470;            // wait after last stroke before recognizing
const SET_LEN = 22;            // problems per set
const LIMIT_MS = 5 * 60 * 1000; // 5 minutes
const MAX_N = SET_LEN - 1;     // cap back-level so a set stays well-formed
const LEVEL_KEY = "onikeisan.level";
const BEST_KEY = "onikeisan.best";   // best = highest back-level reached

let level = clampLevel(+localStorage.getItem(LEVEL_KEY) || 1);
let best = Math.max(1, +localStorage.getItem(BEST_KEY) || 1);
let playing = false, locked = false, phase = "answer"; // "memorize" | "answer"
let startMs = 0, clockTick = null;
let set = null;                 // { probs:[{a,b,op,ans}], t, correct }
let maxLevelThisPlay = 1;
let hasInk = false, recogTimer = null;

bestEl.textContent = best + "バック";
levelEl.textContent = level + "バック";

function clampLevel(n) { return Math.max(1, Math.min(MAX_N, n | 0)); }

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
function clearPad() { ctx.clearRect(0, 0, pad.width, pad.height); hasInk = false; }
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
  if (locked || !playing || phase !== "answer") return;
  e.preventDefault();
  if (recogTimer) { clearTimeout(recogTimer); recogTimer = null; }
  drawing = true;
  const p = posOf(e); lastX = p.x; lastY = p.y;
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.strokeStyle = "#f4f6f8"; ctx.lineWidth = strokeW;
  ctx.beginPath(); ctx.arc(lastX, lastY, strokeW / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#f4f6f8"; ctx.fill();
  hasInk = true;
}
function strokeMove(e) {
  if (!drawing) return;
  e.preventDefault();
  const p = posOf(e);
  ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke();
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
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      if (data[(y * W + x) * 4 + 3] > 24) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
  if (maxX < 0) return null;
  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const dw = Math.max(1, Math.round(bw * scale)), dh = Math.max(1, Math.round(bh * scale));
  const ox = (28 - dw) / 2, oy = (28 - dh) / 2;
  tctx.clearRect(0, 0, 28, 28);
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(pad, minX, minY, bw, bh, ox, oy, dw, dh);
  const td = tctx.getImageData(0, 0, 28, 28).data;
  const raw = new Float32Array(784);
  let sum = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {
    const v = td[i * 4 + 3] / 255;
    raw[i] = v; sum += v; cx += (i % 28) * v; cy += ((i / 28) | 0) * v;
  }
  if (sum === 0) return null;
  const shiftX = Math.round(14 - cx / sum), shiftY = Math.round(14 - cy / sum);
  if (shiftX === 0 && shiftY === 0) return raw;
  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    const sy = y - shiftY; if (sy < 0 || sy >= 28) continue;
    for (let x = 0; x < 28; x++) {
      const sx = x - shiftX; if (sx < 0 || sx >= 28) continue;
      out[y * 28 + x] = raw[sy * 28 + sx];
    }
  }
  return out;
}

// ---------- problems ----------
function rnd(n) { return Math.floor(Math.random() * n); }
function genProblem() {
  if (Math.random() < 0.5) { const a = rnd(10), b = rnd(10 - a); return { a, b, op: "+", ans: a + b }; }
  const a = rnd(10), b = rnd(a + 1); return { a, b, op: "−", ans: a - b };
}
function newSet() {
  const probs = [];
  for (let i = 0; i < SET_LEN; i++) probs.push(genProblem());
  return { probs, t: 0, correct: 0 };
}

// ---------- turn engine ----------
// turn t (0-based) over a set: show problem[t] (memorize), answer due for
// problem[t-N]. Total turns = SET_LEN + N; answers due on turns N..SET_LEN+N-1.
function renderTurn() {
  const N = level, t = set.t;
  const showIdx = t, ansIdx = t - N;

  if (showIdx < SET_LEN) {
    const p = set.probs[showIdx];
    qaEl.textContent = p.a; opEl.textContent = p.op; qbEl.textContent = p.b;
    problemEl.classList.remove("hidden");
  } else {
    problemEl.classList.add("hidden"); // wind-down: no new problem to memorize
  }

  const answered = Math.max(0, t - N);
  progressFill.style.width = (answered / SET_LEN * 100) + "%";

  if (ansIdx < 0) {
    phase = "memorize";
    padWrap.classList.add("memorize");
    instructEl.classList.add("memo");
    instructEl.textContent = N === 1 ? "おぼえてね" : `おぼえてね（${N}つ あとで こたえる）`;
    nextBtn.style.display = ""; clearBtn.style.display = "none"; passBtn.style.display = "none";
    clearPad();
  } else {
    phase = "answer";
    padWrap.classList.remove("memorize");
    instructEl.classList.remove("memo");
    const ord = N === 1 ? "1つまえ" : `${N}つまえ`;
    instructEl.textContent = `${ord}の こたえ  (${answered + 1}/${SET_LEN})`;
    nextBtn.style.display = "none"; clearBtn.style.display = ""; passBtn.style.display = "";
    clearPad();
  }
}

function advanceTurn() {
  set.t++;
  if (set.t >= SET_LEN + level) endSet();
  else renderTurn();
}

function recognize() {
  recogTimer = null;
  if (locked || !playing || phase !== "answer") return;
  const grid = getInkGrid();
  if (!grid) return;
  const { digit, probs } = predict(grid);
  const expected = set.probs[set.t - level].ans;
  // We know the expected answer, so accept the top prediction or a confident
  // runner-up — reduces unfair misreads while still requiring the right recall.
  const ok = digit === expected || probs[expected] >= 0.34;
  if (ok) set.correct++;
  showVerdict(ok ? expected : digit, ok);
}

function showVerdict(num, ok) {
  locked = true;
  verdictEl.textContent = num;
  verdictEl.className = ""; void verdictEl.offsetWidth;
  verdictEl.classList.add("show", ok ? "ok" : "bad");
  vibrate(ok ? 25 : [0, 25, 45, 25]);
  setTimeout(() => { locked = false; advanceTurn(); }, ok ? 340 : 480);
}
function vibrate(p) { if (navigator.vibrate) try { navigator.vibrate(p); } catch (_) {} }

// ---------- set / level flow ----------
function endSet() {
  const acc = set.correct / SET_LEN; // 22 answers per set
  const pct = Math.round(acc * 100);
  let delta = 0, kind = "stay";
  if (acc >= 0.85 && level < MAX_N) { delta = 1; kind = "up"; }
  else if (acc <= 0.65 && level > 1) { delta = -1; kind = "down"; }
  level = clampLevel(level + delta);
  localStorage.setItem(LEVEL_KEY, level);
  levelEl.textContent = level + "バック";
  if (level > maxLevelThisPlay) maxLevelThisPlay = level;
  if (maxLevelThisPlay > best) { best = maxLevelThisPlay; localStorage.setItem(BEST_KEY, best); bestEl.textContent = best + "バック"; }

  const msg = kind === "up" ? `<span class="up">せいかい率 ${pct}% → ▲ ${level}バックにアップ！</span>`
            : kind === "down" ? `<span class="down">せいかい率 ${pct}% → ▼ ${level}バックにダウン</span>`
            : `せいかい率 ${pct}% → ${level}バックのまま`;

  if (Date.now() - startMs >= LIMIT_MS) {
    toast(msg, () => endGame());
  } else {
    toast(msg, () => { set = newSet(); renderTurn(); });
  }
}

function toast(html, done) {
  locked = true;
  toastEl.innerHTML = html;
  toastEl.classList.add("show");
  setTimeout(() => {
    toastEl.classList.remove("show");
    setTimeout(() => { locked = false; done && done(); }, 220);
  }, 1500);
}

// ---------- game flow ----------
function updateClock() {
  const s = Math.floor((Date.now() - startMs) / 1000);
  timerEl.textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  if (Date.now() - startMs >= LIMIT_MS) timerWrap.classList.add("warn");
}
function startGame() {
  overlay.classList.add("hidden");
  playing = true; locked = false;
  startMs = Date.now();
  maxLevelThisPlay = level;
  timerWrap.classList.remove("warn");
  updateClock();
  if (clockTick) clearInterval(clockTick);
  clockTick = setInterval(updateClock, 500);
  set = newSet();
  renderTurn();
}
function endGame() {
  playing = false;
  clearInterval(clockTick); clockTick = null;
  if (recogTimer) { clearTimeout(recogTimer); recogTimer = null; }
  const newRecord = maxLevelThisPlay >= best && maxLevelThisPlay > 1;
  overlayMsg.innerHTML =
    `このかい の さいこう：<b>${maxLevelThisPlay}バック</b><br>` +
    `つぎは <b>${level}バック</b> からスタート<br>` +
    (newRecord ? `🎉 ベストこうしん！ <b>${best}バック</b>` : `ベスト：<b>${best}バック</b>`);
  startBtn.textContent = "もういちど";
  overlay.classList.remove("hidden");
}

startBtn.addEventListener("click", startGame);
clearBtn.addEventListener("click", () => { if (!locked && phase === "answer") clearPad(); });
nextBtn.addEventListener("click", () => { if (!locked && playing && phase === "memorize") advanceTurn(); });
// pass: give up on this one — show the correct answer (not counted as correct) and move on
passBtn.addEventListener("click", () => {
  if (locked || !playing || phase !== "answer") return;
  if (recogTimer) { clearTimeout(recogTimer); recogTimer = null; }
  showVerdict(set.probs[set.t - level].ans, false);
});

// ---------- boot ----------
resize();
predict(new Float32Array(784)); // warm up so first recognition is instant
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js").catch(() => {}));
}
