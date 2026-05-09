import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from models import get_resnet50_model
from dataset import transform as preprocess


# ==========================================
# 進捗メモ

# 4/23 画像の読み込みグレー統一 & 重なりのコードimportで省略+完全一致に
# 4/24 maxの切り上げ問題 
# 4/25 層の変更 -> 2層がよき
#        maxの問題も画像によって違い確認
# 5/3 base_nameあたりのコード整え、出力先の切り替え
#     いろんな画像で、いろんなブロック試し(出力：Grad_Cam)
#     .abs と.maximumの比較 (出力：Grad_Abs)
#   　結論　2層目で筋管を把握。3,4層はうまくみれない　Grad-Camの役割としては十分なのでは
#          abs見たら一層目でバックグラウンドみてるの分かる
# ==========================================

# ==========================================
# 動かすメモ
# 　初めに、ハイパラ & みる層＋ブロック ＆ 写真のパス を指定
#       出力の形変える以外はもう変更しなくていいようにまとめ済み(5/3)
#   正規化について
#  　  np.maximumで負の関数Relu処理(absも試し済みだがnotメイン)
#   出力
#       Grad_Cam/ep数_lr数 のフォルダに '画像名_layer数_block数.png' で保存 
#
#
#   改善したいことメモ(5/3)
#       コメントが汚い & いらないの消したい
#       GradCam_Xlayerのフォルダたちはabsのやつしばらくしたらいらないから消したい　GradMaxのフォルダもいらない
# ==========================================


# ==========================================
#                 初期設定
# ==========================================
num_epochs = 50
learning_rate = 0.0001
layer_num = 2
block_num = 2
base_name = 'No.3_00308' 
use_img_name = f'{base_name}_Isousa.tif'

# ==========================================
# 1. GradCAMクラスの定義(ヒートマップの計算)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # フックの登録（順伝播の特徴マップと逆伝播の勾配）
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    #指定した層で出力された特徴マップ
    def save_activation(self, module, input, output):
        self.activations = output
    #同じく勾配
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        # 1. 順伝搬 (入力 画像x 出力は予測値の実数　スカラー)
        self.model.zero_grad()
        output = self.model(x)

        # 2. 逆伝播 
        # 回帰なので、予測値(スカラー)そのものが起点
        output.backward()

        # 3. GradCAMの計算
        # プーリング層の勾配の平均を計算（重みwとなる）
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 指定した層での特徴マップを切り取って保存
        activations = self.activations.detach()
        
        # 重みの形を [1, 2048, 1, 1] に変形して、特徴マップ掛け合わせる
        weights = pooled_gradients.view(1, -1, 1, 1)
        weighted_activations = activations * weights
        
        # チャネル方向に合計
        heatmap = torch.sum(weighted_activations, dim=1).squeeze()

        # -------------------------------------
        # 正規化
        # 1. テンソル -> Numpy配列に変換
        # 2. Relu関数で負の値0に (絶対値の.absも試し済み)
        # 3. (上の値)÷(全体の最大値)で0~1に正規化 (一応最大値が0のときのエッジケース考えてif してるけど起こりうることはなさげ -> 消してもよきかも)
        # -------------------------------------
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        # heatmap = np.abs(heatmap)　#5/3 この子はもう使わない
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            print('ヒートマップは最大値で正規化DONE')
        else:
            print('ヒートマップの最大値が0だったから割ってない')    #起こりうる？一応
        
        return heatmap, output.item()

# ==========================================
# 2. ヒートマップと元画像の重ね合わせ(出力用)
# ==========================================
def save_cam_on_image(img_path, heatmap):
    # 1. 元画像読み込み->リサイズ->numpy配列で0~1
    img = Image.open(img_path).convert('L')   #4/22変えたいけどそのあとの参照チャネル数が違う
    img = img.resize((224, 224)) 
    img_np = np.array(img) / 255.0
    img_np = np.stack([img_np, img_np, img_np], axis=-1) # 4/23　上のフレーに合わせてこれ追加した

    # 2. ヒートマップを画像サイズに拡大
    # ヒートマップ 0~1のfloat (ex 7×7) -> 8ビット符号なし整数 -> リサイズ
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)) # 一旦画像にする
    heatmap_img = heatmap_img.resize((224, 224), resample=Image.BILINEAR) # 滑らかに拡大
    
    # 3. ヒートマップに色をつける
    # 8ビット符号なし -> Numpy配列 0~1 に戻す
    heatmap_np_resized = np.array(heatmap_img) / 255.0

    #この時点でヒートマップと元画像は両方Numpy配列0~1に揃った
    
    # 'jet' カラーマップを適用 (RGBAになるのでRGBの3つだけ取る)
    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap_np_resized)[:, :, :3] # shape: (224, 224, 3)

    # 4. ヒートと元画像重ね合わせ
    alpha = 0.4 # ヒートマップの透明度 40%
    superimposed_img = heatmap_colored * alpha + img_np * (1 - alpha)
    
    # 1を超えた値をクリップ（念のため）
    superimposed_img = np.clip(superimposed_img, 0, 1)

    # 5. 表示
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"{base_name}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"layer{layer_num}_block{block_num}")
    plt.axis('off')
    
    # 出力先(ハイパラごとのフォルダ先にした)
    save_dir = f"Grad_Cam/ep{num_epochs}_lr{learning_rate}"
    os.makedirs(save_dir, exist_ok=True)

    # 出力画像名 (層とブロックはファイル名で管理 ->層ごとのフォルダ管理してたけどこっちの方が一旦楽そう)
    out_path = os.path.join(save_dir, f"{base_name}_{layer_num}_{block_num}.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out_path}")


# ==========================================
# 3. メイン処理
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル準備
    model = get_resnet50_model()
    # 学習済み重みをロード
    model.load_state_dict(torch.load(f"output/train/best_Resnet50_ep{num_epochs}_lr{learning_rate}.pth", map_location="cpu"))
    model.to(device)
    model.eval()

    # ターゲット層の指定
    target_layer = getattr(model, f'layer{layer_num}')[block_num]

    # GradCAMのインスタンス化
    grad_cam = GradCAM(model, target_layer)

    # 解析したい画像選択
    img_path = f"data/Isousa/{use_img_name}" 

    try:
        image = Image.open(img_path).convert('L')     #4/23 ここRGB->L
        input_tensor = preprocess(image).unsqueeze(0).to(device)    # preprocessはdataset.py のtransformと同じ

        # GradCAM実行
        heatmap, prediction = grad_cam(input_tensor)
        
        print(f"モデルの予測値: {prediction:.4f}")
        
        # 結果表示
        save_cam_on_image(img_path, heatmap)

    except FileNotFoundError:
        print(f"画像ファイル {img_path} が見つかりません。パスを確認して。")