import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # 進捗バー表示用

# ==========================================
# 1. モデル定義 (頂いたコード)
# ==========================================
def get_resnet50_model(dropout_rate=0.5):
    # 最新のPyTorchでは weights 引数が推奨されていますが、
    # 学習済み重みをロードする場合は初期値は何でも構いません
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1)) 
    return model

# ==========================================
# 2. 特徴量抽出の準備
# ==========================================
def extract_features(model, dataloader, device):
    """
    データローダーから画像を読み込み、モデルに通して特徴量(2048次元)を返す関数
    """
    model.eval() # 評価モード
    features_list = []
    
    with torch.no_grad(): # 勾配計算なし（メモリ節約・高速化）
        for inputs, _ in tqdm(dataloader, desc="Extracting"):
            inputs = inputs.to(device)
            
            # モデルに入力 (fcがIdentityになっているので2048次元が出てくる)
            outputs = model(inputs)
            
            # CPUに戻してnumpy変換しリストに追加
            features_list.append(outputs.cpu().numpy())
            
    # リストを結合して (N, 2048) の行列にする
    return np.concatenate(features_list, axis=0)

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- A. モデルの用意 ---
    model = get_resnet50_model()
    
    # ★重要: ここで学習済みの重みをロードしてください
    model.load_state_dict(torch.load("output_soturonn/train/resnet50_fold1_ep50_lr0.001.pth", map_location="cpu"))
    print("学習済み重みをロードしました")

    # --- B. モデルの改造 (2048次元を取り出す) ---
    # 最後の回帰層(Dropout+Linear)を無効化し、スルーさせる
    model.fc = nn.Identity()
    model = model.to(device)

    # --- C. データの用意 (ここはご自身のDataLoaderに置き換えてください) ---
    # ※ここではダミーデータを作成しています
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.230, 0.230, 0.230], std=[0.097, 0.097, 0.097])
    ])
    
    # 仮: Train 200枚, Test 200枚 (本来はImageFolderなどで読み込みます)
    # バッチサイズはGPUメモリに合わせて調整 (例: 32, 64)
    dummy_train = torch.randn(758, 3, 224, 224) 
    dummy_test = torch.randn(163, 3, 224, 224)
    
    train_loader = DataLoader(TensorDataset(dummy_train, torch.zeros(758)), batch_size=32)
    test_loader = DataLoader(TensorDataset(dummy_test, torch.zeros(163)), batch_size=32)

    # --- D. 特徴量の抽出実行 ---
    print("Trainデータの特徴抽出中...")
    X_train = extract_features(model, train_loader, device)
    
    print("Testデータの特徴抽出中...")
    X_test = extract_features(model, test_loader, device)

    print(f"Train形状: {X_train.shape}, Test形状: {X_test.shape}")
    # 例: (200, 2048), (200, 2048) になるはず

    # --- E. データの結合とラベル付け ---
    # Train=0, Test=1 としてラベルを作成
    X = np.vstack([X_train, X_test])
    y = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])

    # --- F. 可視化 (PCA & t-SNE) ---
    # まず標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # t-SNE (データ数が多い場合は時間がかかります)
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def plot_scatter(ax, X_2d, title):
        ax.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='blue', label='Train', alpha=0.5, s=20)
        ax.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='red', label='Test', alpha=0.5, s=20)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_scatter(axes[0], X_pca, "PCA: Train(Blue) vs Test(Red)")
    plot_scatter(axes[1], X_tsne, "t-SNE: Train(Blue) vs Test(Red)")
    
    plt.tight_layout()
    out_path = "output_soturonn/PCA_tSNE_resnet50_fold1.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out_path}")

if __name__ == "__main__":
    main()