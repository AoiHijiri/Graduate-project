# Handwritten Math Solver（初學者機器學習練習）

這是一個**給機器學習初學者閱讀與修改**的迷你專案：它用一個不需要外部資料集、不需要 GPU、也不需要大型框架的 k-nearest neighbours（kNN）模型，辨識單行手寫風格數學式中的字元，接著把辨識出的式子交給解題引擎計算。

> 重要提醒：這是教學型 demo，不是成熟 OCR 產品。真實手寫數學辨識通常需要 CNN/Transformer、資料標註、版面分析，以及大量訓練資料。這個專案刻意保持小而完整，方便你理解「資料 → 訓練 → 預測 → 解題」的流程。

## 功能

- 內建合成訓練資料：從 5x7 字元樣板產生帶有位移與雜訊的「手寫風格」樣本。
- 使用簡單 kNN 模型辨識：
  - 數字 `0-9`
  - 未知數與常用變數：`x`, `y`, `z`, `k`
  - 算術符號：`+`, `-`, `*`, `/`, `=`, `^`, `(`, `)`, `,`
  - 常見常數：`p` 會轉成 `pi`，`e` 代表自然常數
  - 進階符號別名：`S` 代表 `sigma`，`I` 代表 `integral`
- 解題：
  - 沒安裝 SymPy 時，可計算基本四則運算與 `pi`/`e`。
  - 安裝 SymPy 後，可解方程、積分、sigma summation 與符號化簡。

## 專案結構

```text
src/handwritten_math_solver/
  symbols.py       # 5x7 字元樣板與符號別名
  model.py         # 合成資料與 kNN 模型
  ocr.py           # PGM/ASCII 圖像讀取、字元切割、辨識流程
  math_engine.py   # 正規化與解題引擎（SymPy 或安全 fallback）
  cli.py           # 命令列入口
```

## 快速開始

### 1. 直接計算文字算式

不安裝任何套件也可以先跑基本算術：

```bash
PYTHONPATH=src python -m handwritten_math_solver.cli --text "2+3*4"
```

輸出會類似：

```text
recognized: 2+3*4
normalized: 2+3*4
answer: 14.0
engine: safe-eval
```

### 2. 安裝 SymPy 以支援解方程、積分與 sigma

```bash
python -m pip install -r requirements.txt
```

範例：

```bash
PYTHONPATH=src python -m handwritten_math_solver.cli --text "2*x+3=7"
PYTHONPATH=src python -m handwritten_math_solver.cli --text "integral(x^2,x)"
PYTHONPATH=src python -m handwritten_math_solver.cli --text "sigma(k^2,k,1,5)"
```

### 3. 辨識簡單的「手寫風格」圖像

此 demo 支援兩種無額外依賴的輸入格式：

1. ASCII art：用 `#`、`1`、`X` 或 `@` 表示黑色筆畫。
2. PGM（P2/P5）灰階圖片：若你有 PNG/JPG，可用 ImageMagick 轉檔，例如 `magick input.png output.pgm`。

PGM 範例：

```bash
PYTHONPATH=src python -m handwritten_math_solver.cli --pgm expression.pgm
```

## 學習路線建議

1. 先看 `symbols.py`：理解每個字元如何被表示成 5x7 二值圖。
2. 再看 `model.py`：觀察如何把樣板加入雜訊與位移，產生訓練資料。
3. 修改 `samples_per_class`、`noise` 或 `k`，觀察辨識結果如何改變。
4. 嘗試新增符號，例如 `sin`、`cos` 或更多希臘字母。
5. 若想升級成真正的手寫辨識，可把 kNN 換成 CNN，並使用 CROHME 或 EMNIST 類型的資料集。

## 測試

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m compileall src tests
```
