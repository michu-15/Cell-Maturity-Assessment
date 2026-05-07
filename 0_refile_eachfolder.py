import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm  # 最初にインポート

# =================設定エリア=================
# データが入っている親フォルダ（「No.xx」の場所）
SOURCE_DIR = r"moto_data/No.6"

# 整理後のデータを保存するフォルダ（新しく作られます）
OUTPUT_DIR = SOURCE_DIR

# 対象のチャンネル名と、作成するフォルダ名の対応
# キー：ファイル名に含まれる文字, 値：作成するフォルダ名
CHANNEL_MAP = {
    "CH1": "DAPI",
    "CH2": "MHC",
    "CH4": "Isousa",
    "Overlay": "Merge",
}
# ===========================================

def organize_images():
    # 保存先フォルダが存在しない場合は作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 保存先フォルダを作成しました: {OUTPUT_DIR}")

    # 各チャンネルのフォルダを作成
    for folder_name in CHANNEL_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)

    device_folder = Path(SOURCE_DIR)
    device_name = device_folder.name

    processed_count = 0

    print("🚀 処理を開始します...")



    # フォルダの中の画像ファイルを取得 (拡張子はjpg, png, tifなどに対応)
    for file_path in tqdm(device_folder.iterdir()):
        if file_path.is_dir(): 
            continue # フォルダは無視

        file_name = file_path.name # 例: Image_00113_CH1.tif
        
        # どのチャンネルに該当するかチェック
        target_channel = None
        for key, folder_name in CHANNEL_MAP.items():
            if key in file_name:
                target_channel = folder_name
                break
        
        # CH1, CH2, CH3, merge のいずれにも該当しないファイルはスキップ
        if target_channel is None:
            continue

        # --- 画像番号の抽出 ---
        # 想定: Image_00113_CH1.tif → "_" で分割して番号を取り出す
        # file_name_stem は拡張子なしの名前 (Image_00113_CH1)
        parts = file_path.stem.split('_') 
        
        # 画像番号を探す
        # 数字だけで構成されているか
        img_number = "00000"
        for part in parts:
            if part.isdigit():
                img_number = part
                break
        
        # --- 新しいファイル名の作成 ---
        # new_name: No.〇〇_画像番号_フォルダ名.拡張子
        # 例: No.01_00113_CH1.tif
        extension = file_path.suffix # .tif や .png
        new_filename = f"{device_name}_{img_number}_{target_channel}{extension}"
        
        # --- コピー実行 ---
        dest_path = os.path.join(OUTPUT_DIR, target_channel, new_filename)
        
        try:
            shutil.copy2(file_path, dest_path) # copy2でメタデータも保持
            #print(f"処理中: {device_name} | {file_name} -> {target_channel}/{new_filename}")
            processed_count += 1
        
        
        except Exception as e:
            print(f"エラー発生: {file_name} - {e}")

    print("-" * 30)
    print(f"✨ 完了しました！ 合計 {processed_count} 枚の画像を整理しました。")
    print(f"保存先: {OUTPUT_DIR}")

if __name__ == "__main__":
    organize_images()