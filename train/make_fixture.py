"""Rebuild the model from the EXPORTED weights (docs/model.js) and dump a
fixture of MNIST test images + torch predictions, so we can verify the
pure-JS forward pass produces identical results."""
import base64, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
js = open(os.path.join(HERE, "..", "docs", "model.js")).read()
blob = js.split("export const MODEL =", 1)[1].strip().rstrip(";\n").rstrip(";")
M = json.loads(blob)

def t(name):
    p = M[name]
    arr = np.frombuffer(base64.b64decode(p["b64"]), dtype="<f4").reshape(p["shape"]).copy()
    return torch.from_numpy(arr)

class TinyCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        s.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        s.fc = nn.Linear(16 * 7 * 7, 10)
    def forward(s, x):
        x = F.max_pool2d(F.relu(s.conv1(x)), 2)
        x = F.max_pool2d(F.relu(s.conv2(x)), 2)
        return s.fc(x.flatten(1))

m = TinyCNN()
m.load_state_dict({k: t(k) for k in ["conv1.weight","conv1.bias","conv2.weight","conv2.bias","fc.weight","fc.bias"]})
m.eval()

ds = datasets.MNIST(os.path.join(HERE, "data"), train=False, download=True, transform=transforms.ToTensor())
N = 50
imgs, preds = [], []
correct = 0
with torch.no_grad():
    for i in range(N):
        x, y = ds[i]
        logit = m(x.unsqueeze(0))[0]
        pred = int(logit.argmax())
        imgs.append([round(float(v), 6) for v in x.flatten().tolist()])
        preds.append(pred)
        correct += (pred == y)
out = os.path.join(HERE, "fixture.json")
json.dump({"imgs": imgs, "preds": preds}, open(out, "w"))
print(f"wrote {out}  N={N}  torch_acc={correct/N:.2f}")
