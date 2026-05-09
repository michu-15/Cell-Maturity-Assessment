from pathlib import Path
import pandas as pd
import shutil
import re

# ==========================================
# 動かすメモ
# 【役割】
    # ①全スコアのcsvまとめ　data/merged_all_label.py
    # ②各No.フォルダにある位相差画像のまとめ data/Isousa
# 【入力】
    # csv: moto_data直下に 'No.XX_score.csv' がいくつもある
    #         -> 使いたくないNo.のスコアcsvはスキップ可能(指定したNo.XX の除外)
    # Isousa画像: moto_data/各No.XX/Isousa があるはず．これらをまとめる
# 【出力】
    # data フォルダを新しく生成
    #     ->その中にそれぞれ data/isousa と data/merged_all_score.csv が生成

# 【OldとNewのcsvのカラム名 5/7最新】
    # Old_csv: Image Name,DAPI File Used,MHC File Used,Maturity score,DAPI Count,Merged Count
    # new_csv : Image Name,Maturity score


# どっちか片方だけやり直したいならコメントアウト
# 画像のまとめは，連チャンでやるとき用にしてないからフォルダごと消す
#     -> 空にするor上書きかはそのときによるから一旦追記はなし

# ==========================================

# ==========================================
# 進捗メモ
    # 卒論まで :分類問題にするときののクラスタリング用(3クラス)コードもここで処理
    # 26年 GWまでのチェック : 分類コード削除 
    #                         -> soturonn_secoundのフォルダに元データはあり．
    #                         -> 元コードは綺麗にしてない．戻るならここと比較
    # 5/7 44行目の DATA_DIR.mkdir(parents=True, exist_ok=True) parents=Trueは中間層作成のため -> あとで消す
    # あとやりたいこと(5/7)
    #     いらないコメント削除
# ==========================================



# ==========================
# 設定
# ==========================
MOTO_DIR = Path("../tus_ishi/moto_data")   # No.XX_score.csv がある場所
DATA_DIR = Path("../tus_ishi/data")        # 出力先
EXCLUDE_NOS = []        # ← 除外したい No.XX を書く (卒論はNo.1,8を除外)
OUT_CSV = DATA_DIR / "merged_all_score.csv"


# csvのファイル名の指定 
# moto_dataの直下にあって、'No.XX_score.csv' という命名になっている前提
def is_no_xx_score(p: Path) -> bool:
    return re.fullmatch(r"No\.\d+_score\.csv", p.name) is not None


def main():
    # DATA_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================
    # 1. CSV結合
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
        required = {"DAPI File Used", "Maturity score"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} に必須カラムないよ〜")

        #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
        # 4/13 追加
        #　以下の部分は明日のCellMTGに合わせるために勝手にかえた
        # 上のrequired も変える

        #　あとで変更・修正必須　もとのコードは以下
        # df = df[["Image Name", "Maturity score"]].copy() 
        # df["Maturity score"] = pd.to_numeric(df["Maturity score"], errors="coerce") 
        # df['Image Name'] = df['Image Name'].str.replace('DAPI', 'Isousa')


        # 5/7追記
        # そもそも base_name = 'Image Name' のカラムで 拡張子なしのベースパスを作った。このあとかえてもいいえけど
        # したので動いてるならいったんこれでおけ
        # かえるとしたらbase_name + Isousa.tif　の形だけどreplaceでもきれいさ変わらないかも　じゃあFijiでbase_nameのカラムそもそもいらない説もある
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
    

    merged.to_csv(OUT_CSV, index=False)
    print(f"全スコアまとめたCSVの保存先 -> {OUT_CSV}")



    # ==========================
    # 2. Isousa画像を移動
    # ==========================
    isousa_out = DATA_DIR / "Isousa"
    isousa_out.mkdir(exist_ok=True)

    for sub in MOTO_DIR.iterdir():
        if not sub.is_dir():
            continue
        if not re.fullmatch(r"No\.\d+", sub.name):
            continue
        if sub.name in EXCLUDE_NOS:
            continue

        src_dir = sub / "Isousa"
        if not src_dir.exists():
            continue

        for img in src_dir.iterdir():
            if img.is_file():
                dst = isousa_out / img.name
                if dst.exists():
                    raise FileExistsError(f"同名ファイルあり: {dst}")
                shutil.copy2(img, dst)

    print(f"次の場所にIsousa移動させたよん -> {isousa_out}")
    image_list = list(isousa_out.glob("*")) 
    print(f"位相差画像の総枚数: {len(image_list)}枚")


if __name__ == "__main__":
    main()
