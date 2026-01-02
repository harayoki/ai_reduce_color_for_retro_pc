# MSX Filter Toolkit (minimal)

このパッケージは `spec.md` の最小構成に合わせて、色味変換とドット寄せのステージを備えたツールキットです。CLI とコアロジックのみを用意し、Gradio UI は `ui.py` で後から拡張できるよう空のプレースホルダを置いています。

## インストール（開発）
```bash
pip install -e .
```

## 使い方（例）
```bash
msxfilter --mode color --input in.png --output out.png --palette palettes/msx16.txt
msxfilter --mode pixel --input in.png --output out.png --palette palettes/msx95.txt
msxfilter --mode full --input in.png --output_dir out_dir --palette palettes/msx95.txt
msxfilter --mode sweep --input in.png --output_dir out_dir --palette palettes/msx16.txt --variants 8
```

`palettes/` 配下に MSX16 と MSX95 のサンプルパレットを同梱しています。
