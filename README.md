# 鬼計算 (onikeisann)

スマホで遊ぶ、手書き入力のかんたん計算ドリル。

- 足し算・引き算のみ（1桁どうし、**答えは必ず 0〜9 の1桁**）
- こたえを**指で手書き** → ブラウザ内のミニ AI が即認識 → 自動で次の問題
- **おにモード**: 60秒で何問解けるか（ベストスコアを保存）／ **れんしゅう**: 時間無制限
- 完全オフライン・ライブラリ不要・データはブラウザ(localStorage)に保存

## しくみ

手書き数字認識は、MNIST で学習した**極小 CNN（約9千パラメータ）**の重みを JS に埋め込み、
**純 JavaScript で推論**しています（TensorFlow.js 等は不使用）。答えが必ず 1 桁なので、
0〜9 の1文字だけ判定すればよく、外部通信ゼロで 1ms 未満の認識が可能です。

| ファイル | 役割 |
|---|---|
| `docs/index.html` / `styles.css` | UI（モバイル前提） |
| `docs/app.js` | ゲーム進行・キャンバス手書き・前処理(MNIST互換の正規化) |
| `docs/infer.js` | 純 JS の CNN 順伝播（推論） |
| `docs/model.js` | 学習済みの重み（base64 float32, 自動生成） |
| `docs/sw.js` / `manifest.webmanifest` | PWA（オフライン・ホーム追加） |
| `train/train.py` | モデル学習＆ `docs/model.js` 出力 |

## ローカルで動かす

```sh
cd docs
python3 -m http.server 8000
# ブラウザで http://localhost:8000 （同じ Wi-Fi のスマホからは http://<PCのIP>:8000）
```

## モデルを再学習する

```sh
pip install torch torchvision
python3 train/train.py          # docs/model.js を再生成
python3 train/make_fixture.py   # 検証用フィクスチャ生成（任意）
```

## デプロイ（GitHub Pages）

このリポジトリは `main` ブランチの `/docs` を公開する設定です。`docs/` を更新して push すれば反映されます。
公開 URL: https://mryutaro.github.io/onikeisann/
