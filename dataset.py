import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def each_img_zscore(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + 1e-6)

# 1. 前処理（Transform）を定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # リサイズ
    transforms.ToTensor(),          # 画像をPyTorchが扱える形に変換
#-------------------------------------------------------------------------------
#Merge 使うときは消す
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 1ch -> 3ch
#-------------------------------------------------------------------------------
    transforms.Lambda(each_img_zscore)
])

#2. Dataset を定義

#引数２ 戻り値3(img, score, image_name)
class CellDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        img_name = row['Image Name']      
        score = row['Maturity score']

        #パス作成　：　パス名 = フォルダ名＋img_name
        img_path = os.path.join(self.data_dir, img_name)
        #画像を開く(モノクロ)
        image = Image.open(img_path).convert('L')
        #　Ｍｅｒｇｅの読み込んだ時用（RGB）
        # image = Image.open(img_path).convert('RGB')

        # 4. 画像の前処理（リサイズやTensor変換）
        if self.transform:
            image = self.transform(image)
        
        # 5. スコアをTensor型(float32)に変換    ただのscoreでも動くか確認
        return image, torch.tensor(score, dtype=torch.float32), img_path
