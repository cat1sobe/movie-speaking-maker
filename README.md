# VOICEVOX 音声生成スクリプト

このスクリプトは、テキスト原稿をVOICEVOXを使用して音声ファイルに変換するためのツールです。

## セットアップ

1. VOICEVOXエンジン（Docker）のセットアップ
   ```bash
   # VOICEVOXエンジンの起動
   docker-compose up -d
   ```

2. Pythonパッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. VOICEVOXエンジンの起動確認
   - `docker-compose up -d` でVOICEVOXエンジンが起動していることを確認
   - 初回起動時はイメージのダウンロードに時間がかかります

2. 原稿ファイルの準備
   - `script.txt`を作成
   - UTF-8エンコーディングで保存
   - カスタム間隔マーカーを使用して間隔を指定

3. 音声生成の実行
   ```bash
   python voicevox_generator.py
   ```

## 原稿の書き方

### 基本的な書き方
- UTF-8エンコーディングで保存
- 通常の文章を記述

### カスタム間隔の指定
間隔を指定したい箇所に`<数字s>`の形式でマーカーを挿入します：

```
こんにちは。<3s>
これは3秒の間隔が空きます。<1s>
これは1秒の間隔が空きます。<0.5s>
最後の文章です。
```

- `<3s>`: 3秒の間隔
- `<1s>`: 1秒の間隔
- `<0.5s>`: 0.5秒の間隔（小数点も使用可能）

## 出力
- `voice`ディレクトリに最終的な音声ファイル`final_audio.wav`が生成されます
- 一時ファイルは自動的に削除されます

## 音声調整パラメータ

スクリプトでは以下のパラメータを調整できます：

- 話速（speedScale）: 0.8 - 1.2の範囲で調整可能
- 文末の母音の長さ（vowel_length）: 通常1.0、疑問文で1.2
- 開始前の無音時間（prePhonemeLength）: デフォルト0.1秒
- 終了後の無音時間（postPhonemeLength）: デフォルト0.8秒

## エラー対応

1. VOICEVOXエンジン未起動の場合
   ```
   VOICEVOXエンジンに接続できません。
   1. docker-compose up -d を実行してVOICEVOXエンジンを起動してください。
   2. エンジンの起動完了を待ってから再度実行してください。
   ```

2. 原稿ファイルが見つからない場合
   ```
   原稿ファイル script.txt が見つかりません。
   ```

3. 文字コードエラーの場合
   ```
   原稿ファイル script.txt の文字コードがUTF-8ではありません。
   ```

## 注意事項

- VOICEVOXエンジンは常時起動しておく必要があります
- 原稿ファイルは必ずUTF-8エンコーディングで保存してください
- 大量のリクエストを送信する場合は、適切な待機時間を設定してください
