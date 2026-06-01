// Pure-JS inference for the tiny MNIST CNN trained in train/train.py.
// No external ML library. Decodes base64 float32 weights and runs a
// forward pass on a single 28x28 grayscale image (ink in [0,1]).
import { MODEL } from "./model.js";

function decode(p) {
  const bin = atob(p.b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(buf); // little-endian on all common platforms
}

const W = {
  conv1_w: decode(MODEL["conv1.weight"]), // [8,1,3,3]
  conv1_b: decode(MODEL["conv1.bias"]),   // [8]
  conv2_w: decode(MODEL["conv2.weight"]), // [16,8,3,3]
  conv2_b: decode(MODEL["conv2.bias"]),   // [16]
  fc_w: decode(MODEL["fc.weight"]),       // [10,784]
  fc_b: decode(MODEL["fc.bias"]),         // [10]
};

// 3x3 conv, stride 1, pad 1, followed by ReLU. in: [Cin,H,W] -> out: [Cout,H,W]
function conv3x3relu(input, Cin, H, Wd, weight, bias, Cout) {
  const out = new Float32Array(Cout * H * Wd);
  for (let oc = 0; oc < Cout; oc++) {
    const b = bias[oc];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < Wd; x++) {
        let sum = b;
        for (let ic = 0; ic < Cin; ic++) {
          const inBase = ic * H * Wd;
          const wBase = ((oc * Cin + ic) * 3) * 3;
          for (let ky = 0; ky < 3; ky++) {
            const iy = y + ky - 1;
            if (iy < 0 || iy >= H) continue;
            for (let kx = 0; kx < 3; kx++) {
              const ix = x + kx - 1;
              if (ix < 0 || ix >= Wd) continue;
              sum += input[inBase + iy * Wd + ix] * weight[wBase + ky * 3 + kx];
            }
          }
        }
        out[oc * H * Wd + y * Wd + x] = sum > 0 ? sum : 0;
      }
    }
  }
  return out;
}

// 2x2 max pool, stride 2. in: [C,H,W] -> out: [C,H/2,W/2]
function maxpool2(input, C, H, Wd) {
  const oH = H >> 1, oW = Wd >> 1;
  const out = new Float32Array(C * oH * oW);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < oH; y++) {
      for (let x = 0; x < oW; x++) {
        const iy = y * 2, ix = x * 2;
        const base = c * H * Wd;
        let m = input[base + iy * Wd + ix];
        m = Math.max(m, input[base + iy * Wd + ix + 1]);
        m = Math.max(m, input[base + (iy + 1) * Wd + ix]);
        m = Math.max(m, input[base + (iy + 1) * Wd + ix + 1]);
        out[c * oH * oW + y * oW + x] = m;
      }
    }
  }
  return out;
}

// img: Float32Array(784) ink in [0,1]. Returns { digit, conf, probs }.
export function predict(img) {
  let x = conv3x3relu(img, 1, 28, 28, W.conv1_w, W.conv1_b, 8); // 8x28x28
  x = maxpool2(x, 8, 28, 28);                                   // 8x14x14
  x = conv3x3relu(x, 8, 14, 14, W.conv2_w, W.conv2_b, 16);      // 16x14x14
  x = maxpool2(x, 16, 14, 14);                                  // 16x7x7  (=784)

  const logits = new Float32Array(10);
  for (let o = 0; o < 10; o++) {
    let s = W.fc_b[o];
    const wBase = o * 784;
    for (let i = 0; i < 784; i++) s += W.fc_w[wBase + i] * x[i];
    logits[o] = s;
  }

  // softmax
  let mx = -Infinity;
  for (let i = 0; i < 10; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const probs = new Float32Array(10);
  for (let i = 0; i < 10; i++) { probs[i] = Math.exp(logits[i] - mx); sum += probs[i]; }
  let digit = 0;
  for (let i = 0; i < 10; i++) { probs[i] /= sum; if (probs[i] > probs[digit]) digit = i; }
  return { digit, conf: probs[digit], probs };
}
