import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#------------------------------------------------------

# 画像 & CSV データを train/val/test　の3つに分ける。
# 入力 : csv と全Isousaがまとまったフォルダ
# 出力 : dataフォルダの中に train,val,testの3フォルダ　各フォルダに画像とcsv
# -----------------------------------------------------


def main():
    ###　ここMerge か位相差か確認！
    all_img_dir = "../soturonn_second/data/Isousa"
    all_csv_path = 'soturon_retry/data/merged_all_score.csv'    ###ここも変わってる！
    output_root = "soturon_retry/data_spilt"

    if not os.path.exists(all_csv_path):
            print(f"エラー: CSVファイルが見つかりません -> {all_csv_path}")
            return

    # CSV読み込み
    score_df = pd.read_csv(all_csv_path, header=0)
    print(f"元データ数: {len(score_df)}")

    

    # -----------------------------------------------------
    # 3. データ分割 (Train:70%, Val:15%, Test:15%)
    # -----------------------------------------------------

    # 「テスト用」の切り出し
    train_val_df, test_df = train_test_split(score_df, test_size=0.15, random_state=42)
    # 残りを「学習」と「検証」に分ける
    train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42)

    print(f"学習用: {len(train_df)}, 検証用: {len(val_df)}, テスト用: {len(test_df)} ")

    # -----------------------------------------------------
    # 4. フォルダ作成と画像のコピー
    # -----------------------------------------------------
    # 処理をまとめた関数
    # 引数2 (さっき流さを指定した各df, 見てる名前(trainとか))


    def process_dataset(subset_df, new_folder_name):
        # 保存先フォルダ: data/train, data/val, data/test
        save_dir = os.path.join(output_root, new_folder_name)
        
        # フォルダがなければ作る
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"--- {new_folder_name} データの作成中... ({save_dir}) ---")
        
        # 画像を1枚ずつコピー
        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df)):     # tqdm プログレスバー表示
            img_name = row['Image Name'] 
            
            # 元の画像のパス
            original_path = os.path.join(all_img_dir, img_name)
            # コピー先のパス
            new_path = os.path.join(save_dir, img_name)
            
            try:
                shutil.copy2(original_path, new_path) # copy2
            except FileNotFoundError:
                print(f"警告: 画像が見つかりません {original_path}")
        
        # 分割したCSVをそのフォルダの中に保存 
        csv_save_path = os.path.join(save_dir, f"{new_folder_name}.csv")
        subset_df.to_csv(csv_save_path, index=False)
        print(f"CSV保存完了: {csv_save_path}")

    # 実行
    process_dataset(train_df, "train")
    process_dataset(val_df, "val")
    process_dataset(test_df, "test")

    print("\nすべての処理が完了しました！")


if __name__ == "__main__":
    main()