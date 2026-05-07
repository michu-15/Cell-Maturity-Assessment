import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# 自作ファイルの読み込み
from dataset import CellDataset, transform
from models import get_efficientnet_v2_model, get_convnext_model, get_resnet50_model, get_resnext50_model, get_wide_resnet50_model


# =====================================================
# ▼▼▼ 設定エリア ▼▼▼
# =====================================================

NUM_EPOCH = 50
LEARNING_RATE = 0.00005

# テストデータのパス
TEST_CSV_PATH = "soturon_retry/data_spilt/test/test.csv"
TEST_IMG_DIR = "soturon_retry/data_spilt/test"
TRAIN_RESULT_DIR = "soturon_retry/output/train"
OUTPUT_DIR = "soturon_retry/output/test"

# =====================================================

def get_model_config(model_key):
    """
    モデルごとの設定（構築関数、表示名、グラフの色）を返す関数
    """
    if model_key == 'ENV2':
        return {
            "func": get_efficientnet_v2_model,
            "name": "EfficientNetV2",
            "color": "blue"
        }
    elif model_key == 'Conv':
        return {
            "func": get_convnext_model,
            "name": "ConvNeXt_Tiny",
            "color": "green"
        }
    elif model_key == 'Resnet50':
        return {
            "func": get_resnet50_model,
            "name": "Resnet50",
            "color": "green"
        }
    else:
        raise ValueError(f"未対応のモデルキーです: {model_key}")


def run_evaluation(target_model_key, test_loader, device):
    """
    指定されたモデルで推論・評価・保存を行う関数
    """
    # -----------------------------------------------------
    # 1. モデル設定の取得とパス生成
    # -----------------------------------------------------
    config = get_model_config(target_model_key)
    
    model_func = config["func"]
    model_disp_name = config["name"]
    plot_color = config["color"]
    
    # ★ ファイル名の修正: best_モデル名_ep{}_lr{}.pth
    weight_filename = f"best_{target_model_key}_ep{NUM_EPOCH}_lr{LEARNING_RATE}.pth"
    weight_path = os.path.join(TRAIN_RESULT_DIR, weight_filename)
    
    print(f"\n[{model_disp_name}] 評価開始")
    print(f"読み込む重み: {weight_path}")

    # 保存用ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------
    # 2. モデルのロード
    # -----------------------------------------------------
    model = model_func()
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
    except FileNotFoundError:
        print(f"エラー: 重みファイルが見つかりません -> {weight_path}")
        return

    model.to(device)
    model.eval()

    # -----------------------------------------------------
    # 3. 推論の実行
    # -----------------------------------------------------
    img_names_list = []
    true_scores = []
    preds = []

    print("推論中...")
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, leave=False):
            images = images.to(device)
            output = model(images)
            
            # 結果をリストに追加
            preds.extend(output.cpu().numpy().flatten())
            true_scores.extend(labels.numpy().flatten())
            img_names_list.extend([os.path.basename(p) for p in paths])

    # -----------------------------------------------------
    # 4. スコア計算
    # -----------------------------------------------------
    mae_val = mean_absolute_error(true_scores, preds)
    r2_val = r2_score(true_scores, preds)

    print("-" * 40)
    print(f"モデル: {model_disp_name}")
    print(f"結果  | MAE: {mae_val:.4f} | R2: {r2_val:.4f}")
    print("-" * 40)

    # -----------------------------------------------------
    # 5. 結果の保存 (CSV & Graph)
    # -----------------------------------------------------
    # CSV
    result_df = pd.DataFrame({
        "Image Name": img_names_list,
        "True Score": true_scores,
        "Predicted Score": preds
    })
    csv_filename = f"{target_model_key}_result_ep{NUM_EPOCH}_lr{LEARNING_RATE}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    result_df.to_csv(csv_path, index=False)
    print(f"CSV保存: {csv_path}")

    # グラフ描画
    plt.figure(figsize=(6, 6))
    plt.scatter(true_scores, preds, alpha=0.6, color=plot_color, label='Prediction')
    plt.plot([0, 1], [0, 1], 'r--', label=f'Ideal\nMAE={mae_val:.3f}, $R^2$={r2_val:.3f}')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(f'{model_disp_name} (Ep={NUM_EPOCH}, lr={LEARNING_RATE})')
    plt.xlabel('True Score')
    plt.ylabel('Predicted Score')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    graph_filename = f"{target_model_key}_graph_ep{NUM_EPOCH}_lr{LEARNING_RATE}.png"


    graph_path = os.path.join(OUTPUT_DIR, graph_filename)
    plt.savefig(graph_path)
    plt.close()
    print(f"グラフ保存: {graph_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # データの読み込み (共通)
    if not os.path.exists(TEST_CSV_PATH):
        print("エラー: テストデータが見つかりません。")
        return

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_dataset = CellDataset(test_df, TEST_IMG_DIR, transform=transform)
    # データを一度だけロードして使い回す
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # ==========================================
    #実行したいモデルを指定
    # ==========================================
    
    # EfficientNetV2 を実行したい場合
    # run_evaluation('ENV2', test_loader, device)

    # ConvNeXt を実行したい場合
    # run_evaluation('Conv', test_loader, device)
    run_evaluation('Resnet50', test_loader, device)

if __name__ == "__main__":
    main()