import os
import json
import time
import re
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
from datetime import datetime

class VoicevoxGenerator:
    def __init__(self, host="localhost", port=50021):
        """
        VOICEVOXジェネレーターの初期化
        
        Args:
            host (str): VOICEVOXエンジンのホスト
            port (int): VOICEVOXエンジンのポート
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.sample_rate = 24000  # VOICEVOXのデフォルトサンプリングレート
        
        # VOICEVOXエンジンの接続確認
        try:
            requests.get(self.base_url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "VOICEVOXエンジンに接続できません。\n"
                "1. docker-compose up -d を実行してVOICEVOXエンジンを起動してください。\n"
                "2. エンジンの起動完了を待ってから再度実行してください。"
            )

    def get_speakers(self):
        """
        利用可能な話者の一覧を取得する
        
        Returns:
            list: 話者情報のリスト
        """
        response = requests.get(f"{self.base_url}/speakers")
        return response.json()

    def list_speakers(self):
        """利用可能な話者の一覧を表示する"""
        speakers = self.get_speakers()
        print("\n利用可能な話者一覧:")
        for speaker in speakers:
            for style in speaker["styles"]:
                print(f"話者ID: {style['id']} - {speaker['name']} ({style['name']})")

    def create_audio(self, text, speaker=1, output_path=None):
        """
        テキストから音声を生成する

        Args:
            text (str): 音声に変換するテキスト
            speaker (int): 話者ID
            output_path (str): 出力ファイルパス
        """
        # 音声合成用のクエリを作成
        params = {"text": text, "speaker": speaker}
        query = requests.post(f"{self.base_url}/audio_query", params=params)
        
        if query.status_code != 200:
            raise Exception("音声合成クエリの作成に失敗しました")

        # 音声合成を実行
        synthesis = requests.post(
            f"{self.base_url}/synthesis",
            headers={"Content-Type": "application/json"},
            params={"speaker": speaker},
            data=json.dumps(query.json())
        )

        if synthesis.status_code != 200:
            raise Exception("音声合成に失敗しました")

        # 音声データを保存
        if output_path:
            with open(output_path, "wb") as f:
                f.write(synthesis.content)
            return output_path
        return synthesis.content

    def concatenate_audio_files(self, audio_files):
        """音声ファイルを連結する"""
        if not audio_files:
            return None

        # すべての音声データを読み込んで連結
        combined_audio = []
        for audio_file in audio_files:
            data, sr = sf.read(audio_file)
            if sr != self.sample_rate:
                # サンプリングレートが異なる場合はリサンプリング
                samples = int(len(data) * self.sample_rate / sr)
                data = np.interp(
                    np.linspace(0, len(data), samples, endpoint=False),
                    np.arange(len(data)),
                    data
                )
            combined_audio.extend(data)

        return np.array(combined_audio), self.sample_rate

    def cleanup_temp_files(self, temp_files):
        """一時ファイルを削除する"""
        for file_path in temp_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"警告: 一時ファイルの削除に失敗しました: {file_path} - {str(e)}")

    def process_script(self, script_path, speaker=1, output_dir="voice"):
        """
        原稿ファイルを処理して音声ファイルを生成する

        Args:
            script_path (str): 原稿ファイルのパス
            speaker (int): 話者ID
            output_dir (str): 出力ディレクトリ
        """
        # 出力ディレクトリの作成
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 現在時刻をファイル名に使用
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_path = os.path.join(output_dir, f"voice_{timestamp}.wav")
        temp_dir = os.path.join(output_dir, "temp")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        with open(script_path, "r", encoding="utf-8") as f:
            script = f.read()

        # 間隔マーカーで分割
        segments = re.split(r'<(\d+(?:\.\d+)?s)>', script)
        
        temp_files = []
        
        try:
            for i, segment in enumerate(segments):
                if i % 2 == 0:  # テキストセグメント
                    if segment.strip():
                        current_text = segment.strip()
                        temp_path = os.path.join(temp_dir, f"segment_{i//2}.wav")
                        self.create_audio(current_text, speaker, temp_path)
                        temp_files.append(temp_path)
                else:  # 間隔マーカー
                    try:
                        pause_duration = float(segment.rstrip('s'))
                        # 無音区間を作成（VOICEVOXのサンプリングレートに合わせる）
                        silence = np.zeros(int(self.sample_rate * pause_duration))
                        silence_path = os.path.join(temp_dir, f"silence_{i//2}.wav")
                        sf.write(silence_path, silence, self.sample_rate)
                        temp_files.append(silence_path)
                    except ValueError:
                        print(f"警告: 無効な間隔マーカー: {segment}")

            # 音声ファイルを連結
            if temp_files:
                combined_audio, sample_rate = self.concatenate_audio_files(temp_files)
                sf.write(final_output_path, combined_audio, sample_rate)
                print(f"音声ファイルを生成しました: {final_output_path}")
            else:
                print("警告: 音声ファイルが生成されませんでした")

        finally:
            # 一時ファイルを削除
            self.cleanup_temp_files(temp_files)
            try:
                os.rmdir(temp_dir)
            except:
                print("警告: 一時ディレクトリの削除に失敗しました")

        return final_output_path

def main():
    parser = argparse.ArgumentParser(description="VOICEVOXを使用して音声を生成します")
    parser.add_argument("--script", type=str, help="入力スクリプトファイルのパス")
    parser.add_argument("--speaker", type=int, default=1, help="話者ID（デフォルト: 1）")
    parser.add_argument("--list-speakers", action="store_true", help="利用可能な話者一覧を表示")
    
    args = parser.parse_args()
    
    generator = VoicevoxGenerator()
    
    if args.list_speakers:
        generator.list_speakers()
        return
    
    if not args.script:
        print("エラー: スクリプトファイルを指定してください")
        return
    
    try:
        output_path = generator.process_script(args.script, args.speaker)
        if output_path:
            print(f"音声生成が完了しました: {output_path}")
    except Exception as e:
        print(f"エラー: {str(e)}")

if __name__ == "__main__":
    main() 