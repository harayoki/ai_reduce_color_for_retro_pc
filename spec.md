# MSX Filter Toolkit（最小構成）

目的：
- 画像変換フィルタ（学習なし）
- 2段処理
  - A: 色味変換（16/92色などのパレットへ“寄せる”）
  - B: ドット絵寄せ（8ドット2色などの空間制約へ“近づける”）
- 後で Gradio + Hugging Face Spaces 公開を想定（UI実装はまだ）

設計のキモ：
- “AI”は提案役、**ルール（評価関数 / loss）が主役**
- **最終規格は自作の確定ルーチン**で締める前提（ここでは「だいたい」でOK）
- 色の近さは **Lab（知覚距離）**を土台にする（感覚バイアスは後から追加）
- 「品質しきい値 or 上限回数」で停止（ゼロは狙わない）
- HFモデルを使う場合は **HFキャッシュ共有（HF_HOME等）**を推奨

---

## フォルダ構成（最小）

```text
msxfilter/
  pyproject.toml
  README.md
  src/
    msxfilter/
      __init__.py
      cli.py
      pipeline.py      # A→Bの流れ全部（反復・停止・スイープ含む）
      rules.py         # 評価（Lab/パレット/tile proxy）
      io.py            # 画像・パレットI/O
      ui.py            # Gradio用（後で）
  palettes/
    msx16.txt
    msx95.txt
```

---

## ファイル責務（短く）

### `pipeline.py`
- `run_color_stage()`：色味変換（A）
- `run_pixel_stage()`：ドット寄せ（B、必要なら）
- `run_full()`：A→Bの統合
- 8枚出力などの**パラメータスイープ**
- 停止条件（品質 or 上限回数）

### `rules.py`
- Lab変換（RGB→Lab）
- パレット寄せ評価（soft assign / Lab距離）
- tile proxy（entropy/top-k集中など、必要なら）
- **評価を変える＝ここをいじる**

### `io.py`
- 画像の読込/保存
- パレットtxt（`R G B` or `R,G,B`）読込

### `cli.py`
- CLI入口（例）
  - `msxfilter color --in in.png --out out.png --palette msx92.txt`
  - `msxfilter pixel --in in.png --out out.png --palette msx92.txt`
  - `msxfilter full --in in.png --outdir out/ --palette msx92.txt --variants 8`

### `ui.py`（後で）
- Gradio UI
- `pipeline.py` を呼ぶだけ（UIロジックを本体から分離）

---

## 公開時の表記（控えめに強く）
- 「制約付き画像最適化の考え方（SIGGRAPH/ACM系研究）に着想を得た」
- “再実装”は名乗らない
- 入力画像の権利は利用者が保証（利用規約に一文）
