# 鬼計算 (onikeisann)

スマホで遊ぶ、手書き入力の脳トレ計算。川島隆太教授の「鬼計算」と同じ **N-back 課題** です。

- 計算が1問ずつ出る。**いまの問題は覚えるだけ**で、答えを書くのは **Nつ前**の問題（1バック＝1つ前、2バック＝2つ前…）
- 足し算・引き算のみ（1桁どうし、**答えは必ず 0〜9 の1桁**）
- こたえを**指で手書き** → ブラウザ内のミニ AI が即認識 → 自動で次へ
- **1セット＝22問**。セットの**正解率でレベルが上下**：85%以上→レベルアップ、66〜84%→据え置き、65%以下→ダウン
- 1問ごとに**制限時間**（初期値4秒）。**覚えるフェーズ**は時間切れで自動的に次へ、**答えるフェーズ**は時間切れで**不正解**。パスボタンでスキップ（不正解）も可
- **5分**経過したら、挑戦中のセットを解き終えた時点で終了
- 記録は**到達した最高バック数**（完全オフライン・ライブラリ不要・データは localStorage 保存）

> 認識は、**正しい答えを高確信で書いた瞬間に即判定**（待ち時間ゼロ）。`docs/app.js` の `ANSWER_MS`（答える制限秒）／`MEMO_MS`（覚える制限秒）／`IDLE_MS`（フォールバック待ち）／`INSTANT_CONF`（即判定の確信度）で調整できます。

## しくみ

手書き数字認識は、MNIST で学習した**極小 CNN（約9千パラメータ）**の重みを JS に埋め込み、
**純 JavaScript で推論**しています（TensorFlow.js 等は不使用）。答えが必ず 1 桁なので、
0〜9 の1文字だけ判定すればよく、外部通信ゼロで 1ms 未満の認識が可能です。

| ファイル | 役割 |
|---|---|
| `docs/index.html` / `styles.css` | UI（モバイル前提） |
| `docs/app.js` | ゲーム進行(N-backエンジン)・キャンバス手書き・前処理(MNIST互換の正規化) |
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
