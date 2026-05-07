# models.py
import torch
import torch.nn as nn
import torchvision.models as models

# --- EfficientNet V2 (Small) ---
def get_efficientnet_v2_model(dropout_rate = 0.5):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))  # 回帰用に変更
    return model

# --- ConvNeXt (Tiny) ---
def get_convnext_model(dropout_rate = 0.2):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    
    in_features = model.classifier[2].in_features
    model.classifier[2] =  nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1)) # 回帰用に変更
    return model

# --- ResNet50 (Standard) ---
def get_resnet50_model(dropout_rate=0.5):
    # 最新の重みを使用
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1)) # 回帰用に変更
    return model

# --- ResNeXt50 (32x4d) ---
def get_resnext50_model(dropout_rate=0.5):
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    model = models.resnext50_32x4d(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1)) 
    return model

# --- Wide ResNet50 (Wide ResNet-50-2) ---
def get_wide_resnet50_model(dropout_rate=0.5):
    weights = models.Wide_ResNet50_2_Weights.DEFAULT
    model = models.wide_resnet50_2(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model