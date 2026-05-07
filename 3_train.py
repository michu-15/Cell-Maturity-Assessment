# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time
import os
from tqdm import tqdm
# 他ファイル読み込み
from dataset import CellDataset, transform
from models import get_efficientnet_v2_model, get_convnext_model, get_resnet50_model, get_resnext50_model, get_wide_resnet50_model



# -----------------------------------------------------
# efficientNet もConvNexTもこのコードで
# ハイパーパラメータ： epoch数、学習率、バッチサイズ(l.47,48のDataLoderの中)

# 入力　：　data/ test & train の画像とスコア表
# 出力　：　output/ best_model_〇〇.pth & train_result_〇〇.csv

#-------------切り替えは３箇所-----------------------------

# モデルの選択　 : l. 68,69
# 出力パスの選択 : best_model.pth  -> l.135,136, 
#               train_result.csv -> l.152,152

# 5/7
# 卒論のやり直しでパスの頭に全部soturon_retry がついてる(Dataloderにも)　あとで消す
# アウトプットフォルダのパス指定の場所キモい　Dataloderの上にした　
# 入力のtryもなくしてみていいかも
# 設定値系　全部上に集約したい(実装の落差重視で)
# ----------------------------　-------------------------


def main():

    # -----------------------------------------------------
    # 1. 画像・CSVデータの読み込み
    # -----------------------------------------------------
    try:
        train_df = pd.read_csv("soturon_retry/data_spilt/train/train.csv")
        val_df = pd.read_csv("soturon_retry/data_spilt/val/val.csv")
    except FileNotFoundError:
        print("エラー:data_spilt/train/train.csv が見つからないよ〜")
        return

    print(f"学習データ数: {len(train_df)}, 検証データ数: {len(val_df)}")

    # 結果用のフォルダ作成
    output_dir = "soturon_retry/output/train" ##＃ここも変わってる！！！！！！！！！
    os.makedirs(output_dir, exist_ok=True) 

    # Dataset & DataLoader
    train_dataset = CellDataset(train_df, "soturon_retry/data_spilt/train", transform=transform)
    val_dataset = CellDataset(val_df, "soturon_retry/data_spilt/val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True,  persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # GPU指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 出力地、正規化用の計算用
    min_train = train_df['Maturity score'].min()
    max_train = train_df['Maturity score'].max()
    data_range = max_train - min_train

    min_t = torch.tensor(min_train).float().to(device)
    range_t = torch.tensor(data_range).float().to(device)
    print(f'出力正規化基準値：min;{min_train}, max{max_train}, range = {data_range}')


    # -----------------------------------------------------
    # 3. 学習設定
    # -----------------------------------------------------
    
    num_epochs = 50
    learning_rate = 0.00005
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # モデルの選択
    # model = get_efficientnet_v2_model() 
    # model = get_convnext_model() 
    model = get_resnet50_model()
    # model = get_resnext50_model()
    # model = get_wide_resnet50_model()

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=1e-2) # Adam -> AdamW に変えた decayの設定がいる
    
    # -----------------------------------------------------
    # 3. 学習ループ
    # -----------------------------------------------------
    print(f"Ep {num_epochs}, lr {learning_rate} の学習スタート！")
    #最高記録保存
    best_val_loss = float('inf')
    #損失推移のグラフ用
    train_result = {'train_loss': [], 'val_loss': []}

    start_time = time.time()

    # Early Stopping用　(一旦使わない）
    stop_epoch_count = 10
    no_improve_epoch = 0


    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        running_train_loss = 0.0

        for images, labels, path in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            # ここになんか正則化を入れる-> min-max
            labels_norm = (labels - min_t) / range_t

            
            #勾配りセット
            optimizer.zero_grad()
            #予測
            outputs = model(images)
            #誤差計算
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels_norm)
            #逆伝播
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # === Validation ===
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels, path in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)
                # ここになんか正則化を入れる-> min-max
                labels_norm = (labels - min_t) / range_t

                outputs = model(images)
                loss = criterion(outputs, labels)
                # loss = criterion(outputs, labels_norm)
                running_val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # === 結果の表示と保存 ===
        end_time = time.time()
        elapsed = end_time - start_time

        # 履歴に追加
        train_result['train_loss'].append(epoch_train_loss)
        train_result['val_loss'].append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Time: {elapsed:.1f}sec")

        # モデル保存
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improve_epoch = 0
            # model_file_name = f"best_ENV2_ep{num_epochs}_lr{learning_rate}.pth"
            # model_file_name = f"best_Conv_ep{num_epochs}_lr{learning_rate}.pth"
            model_file_name = f"best_Resnet50_ep{num_epochs}_lr{learning_rate}.pth"
            # model_file_name = f"best_ResNext50_ep{num_epochs}_lr{learning_rate}.pth"
            # model_file_name = f"best_wide50_ep{num_epochs}_lr{learning_rate}.pth"
            save_path = os.path.join(output_dir, model_file_name)
            torch.save(model.state_dict(), save_path)
            print("  -> Best Model Saved!")
        else:
            no_improve_epoch += 1
        
        # Early Stopping用　(一旦使わない）
        # if no_improve_epoch >= stop_epoch_count:
        #     print("学習の停止！")
        #     break



    # -----------------------------------------------------
    # 学習の推移をcsvに保存
    # -----------------------------------------------------

    train_result_df = pd.DataFrame(train_result)
    
    # エポック数の列を追加（1始まりにするため index + 1）
    train_result_df.index.name = 'epoch'
    train_result_df.index = train_result_df.index + 1
    
    # file_name = f"ENV2_ep{num_epochs}_lr_{learning_rate}.csv"
    # file_name = f"Conv_ep{num_epochs}_lr_{learning_rate}.csv"
    file_name = f"Resnet50_ep{num_epochs}_lr{learning_rate}.csv"
    # file_name = f"ResNext50_ep{num_epochs}_lr{learning_rate}.csv"
    # file_name = f"wide50_ep{num_epochs}_lr{learning_rate}.csv"

    csv_path = os.path.join(output_dir, file_name)
    train_result_df.to_csv(csv_path)
    print(f"学習履歴を保存しました: {csv_path}")


if __name__ == "__main__":
    main()