# ＜direnv の venv＞
  * ターミナルでアクティベイト
  -> 上のものだけインタープリターはSSHならなくてまる　
  -> (kawaguchi)の表示になればOkey

  source ~/venvs/kawaguchi/bin/activate
  でアクティベイト
  Interpreter は以下のパスで指定
  /home/mikagucchi/venvs/kawaguchi/bin/python



# moto_data 
  No.3 -> Day3
  No.5 -> Day5 増殖
  No.6 -> Day5 分化

# 0_refile_eachfolder.py
 ->  各No.ごとのファイルにDAPI等のフォルダづくり　DONE

# Fijiでの閾値確認
　　0.01-infinity  auto threshold = mean

# .ijm  の実行＆スコア計算
 　->  おかしいの除く
      　No.3
   ->  New_count_2.ijm  が　うまくまわったぜい！！！！　　　No.3　だけまわしたよん　やったねん

   ## 236からおかしい？ -> 要チェック(No.5のみ検証済み)

# 1_merged_isousa_csv.py
* 以下の部分は一個前で自動的になるように変更 
  ーーーーーーーーーーーーーーーーーーーーーーーーーーー　　
    -> csvファイルを元データの下に移動 
    -> csvファイル名を'No.XX_score.csv'に変更
  ーーーーーーーーーーーーーーーーーーーーーーーーーーー
  ->　全スコアの統合(Image NameにIsousaを付け足し)　＆　位相差画像の統合

   * 4/13 csvの指定カラムの所一旦合わせたので、後で使えるようにもどす
            この時点ではIsousaの移動コメントアウトしてる
    5/7 Isousaもまとめるコードのコメントアウトどっかで解除したい

#  2_spilt_data.py  　の　実行
  -> train/val/test に分ける
  -> 出力 新しい 'data_spilt'　フォルダ
