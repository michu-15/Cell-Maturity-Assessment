from pathlib import Path
import pandas as pd
import shutil
import re


# -----------------------------------------------------
# ①全スコアのcsvまとめ　data/merged_all_label.py
# ②位相差画像のまとめ data/Isousa

# 指定したNo.XX を除外した
# 各csvの結合　＆　分類用の閾値設定とラベル分けしたｃｓｖに変換
# Input : Fiji のカラム名
# 5/7最新 csv: Image Name,DAPI File Used,MHC File Used,Maturity score,DAPI Count,Merged Count
# 　44行目の DATA_DIR.mkdir(parents=True, exist_ok=True) parents=Trueは中間層作成のため　あとで消す
# 分類もある版のカラム名 csv : "Image Name", "Maturity score", "label_class"

# 閾値変えるときは位相差の移動部分をコメントアウト！！！！！！

# -----------------------------------------------------



# ==========================
# 設定
# ==========================
MOTO_DIR = Path("../soturonn_second/moto_data")   # No.XX_score.csv がある場所
DATA_DIR = Path("soturon_retry/data")        # 出力先
EXCLUDE_NOS = []        # ← 除外したい No.XX を書く (卒論はNo.1,8を除外)
OUT_CSV = DATA_DIR / "merged_all_score.csv"

# 3クラス分類（Noneなら自動で1/3, 2/3分位）
THRESH_1 = None
THRESH_2 = None


# csvのファイル名の指定 
# moto_dataの直下にあって、'No.XX_score.csv' という命名になっている前提
def is_no_xx_score(p: Path) -> bool:
    return re.fullmatch(r"No\.\d+_score\.csv", p.name) is not None


def main():
    # DATA_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================
    # 1) CSV結合
    # ==========================
    dfs = []

    for csv_path in sorted(MOTO_DIR.iterdir()):
        if not csv_path.is_file():
            continue
        if not is_no_xx_score(csv_path):
            continue

        no_name = csv_path.stem.replace("_score", "")
        if no_name in EXCLUDE_NOS:
            print(f"Skip: {csv_path.name}")
            continue

        df = pd.read_csv(csv_path)

        # 必須カラム確認
        # required = {"Image Name", "Maturity score"}
        required = {"DAPI File Used", "Maturity score"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} に必須カラムがありません")

        #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
        # 4/13 追加
        #　以下の部分は明日のCellMTGに合わせるために勝手にかえた
        # 上のrequired も変える

        #　あとで変更・修正必須　もとのコードは以下
        # df = df[["Image Name", "Maturity score"]].copy() 
        # df["Maturity score"] = pd.to_numeric(df["Maturity score"], errors="coerce") 
        # df['Image Name'] = df['Image Name'].str.replace('DAPI', 'Isousa')


        # 5/7追記
        # そもそも base_name = 'Image Name' のカラムで 拡張子なしのパスあり。このあとかえてもいいえけど
        # したので動いてるならいったんこれでおけかも
        #------------------------------------------------------
        df = df[["DAPI File Used", "Maturity score"]].copy()
        df = df.rename(columns={"DAPI File Used": "Image Name"})
        df["Maturity score"] = pd.to_numeric(df["Maturity score"], errors="coerce")
        df['Image Name'] = df['Image Name'].str.replace('DAPI', 'Isousa')
        # df['Image Name'] = df['Image Name'] + '_Isousa'

        
        dfs.append(df)
    

    if not dfs:
        raise RuntimeError("結合対象のCSVがありません")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["Maturity score"]).reset_index(drop=True)
    
    # クラス分類するときはこいつもコメントアウト
    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV -> {OUT_CSV}")


    # # ==========================
    # # 2) 3クラス分類
    # # ==========================
    # # 1/3 でいったん
    # if THRESH_1 is None or THRESH_2 is None:
    #     th1 = merged["Maturity score"].quantile(1/3)
    #     th2 = merged["Maturity score"].quantile(2/3)
    #     print(f"[Auto threshold] low={th1:.4f}, high={th2:.4f}")
    # else:
    #     th1, th2 = THRESH_1, THRESH_2

    # def to_class(x):
    #     if x < th1:
    #         return 0
    #     elif x < th2:
    #         return 1
    #     else:
    #         return 2

    # merged["label_class"] = merged["Maturity score"].map(to_class).astype(int)

    # merged = merged[["Image Name", "Maturity score", "label_class"]]
    # merged.to_csv(OUT_CSV, index=False)

    # print(f"Saved CSV -> {OUT_CSV}")
    # # ==========================
    # # 各クラスの枚数確認
    # # ==========================
    # class_counts = merged["label_class"].value_counts().sort_index()

    # print("\n=== label_class counts ===")
    # for cls, cnt in class_counts.items():
    #     print(f"Class {cls}: {cnt}")


    # ==========================
    # 3) Isousa画像を移動
    # ==========================
    # isousa_out = DATA_DIR / "Isousa"
    # isousa_out.mkdir(exist_ok=True)

    # for sub in MOTO_DIR.iterdir():
    #     if not sub.is_dir():
    #         continue
    #     if not re.fullmatch(r"No\.\d+", sub.name):
    #         continue
    #     if sub.name in EXCLUDE_NOS:
    #         continue

    #     src_dir = sub / "Isousa"
    #     if not src_dir.exists():
    #         continue

    #     for img in src_dir.iterdir():
    #         if img.is_file():
    #             dst = isousa_out / img.name
    #             if dst.exists():
    #                 raise FileExistsError(f"同名ファイルあり: {dst}")
    #             shutil.copy2(img, dst)

    # print(f"Moved Isousa images -> {isousa_out}")


if __name__ == "__main__":
    main()
