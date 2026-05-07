# ＜全体の流れ＞
1. data_rename.py ではなく　* refile_eachfolder.py* でDAPI,Isousaにわける
2. Fiji のマクロでそれを解析csvファイルがDAPIのフォルダとかと同じ位置に生成
3. score_csv_cut.py -> リストのDAPIをIsousaに変換 初めの2列（画像名＆スコア）だけ取ってくる。トップにcsvは生成
4. split_data.py でテストとかに分ける　-> split_data のフォルダが作られる
5. train.py で各モデルの学習　-->　best_model.pthが outputにできる
5. train.py 2回！！！
6. ansanmble.py を実行 --> csvと各グラフができる

# ＜学習のし直し＞
train.py & ansanmble.py 両方の エポック数と学習率の変更
train.py は2回実行　（　3箇所変更！！！　　モデルと出力のファイル名　）
ansanmble.py　はそのまま

# ＜direnv の venv＞
source ~/venvs/kawaguchi/bin/activate
でアクティベイト
Interpreter は以下のパスで指定
/home/mikagucchi/venvs/kawaguchi/bin/python


# ＜Python環境の指定＞
右下でPythonのInterpreterを選択　conda を選ぶ

