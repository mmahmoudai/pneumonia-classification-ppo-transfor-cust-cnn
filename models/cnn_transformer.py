import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(CNNBackbone, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.out_channels = base_channels * 8
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        return x


class CNNTransformerClassifier(nn.Module):
    def __init__(self, num_classes=2, img_size=224, base_channels=32, 
                 transformer_dim=256, nhead=4, num_transformer_layers=2, dropout=0.3):
        super(CNNTransformerClassifier, self).__init__()
        
        self.cnn_backbone = CNNBackbone(in_channels=3, base_channels=base_channels)
        
        feature_map_size = img_size // 16
        self.num_patches = feature_map_size * feature_map_size
        
        self.projection = nn.Linear(self.cnn_backbone.out_channels, transformer_dim)
        
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=transformer_dim,
            dropout=dropout
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights (Section 4.3)
        self._init_weights()
    
    def _init_weights(self):
        """
        Weight initialization as per paper Section 4.3:
        - Kaiming normal for convolutional layers
        - Xavier uniform for attention/linear layers
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming normal for convolutional layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier uniform for linear layers (attention projections)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        features = self.cnn_backbone(x)
        
        features = features.flatten(2).transpose(1, 2)
        
        features = self.projection(features)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)
        
        features = self.transformer(features)
        
        cls_output = features[:, 0]
        
        logits = self.classifier(cls_output)
        
        if return_features:
            return logits, cls_output
        return logits
    
    def forward_with_uncertainty(self, x, num_samples=10):
        self.train()
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


def create_model(num_classes=2, img_size=224, base_channels=32, 
                 transformer_dim=256, nhead=4, num_transformer_layers=2, dropout=0.3):
    model = CNNTransformerClassifier(
        num_classes=num_classes,
        img_size=img_size,
        base_channels=base_channels,
        transformer_dim=transformer_dim,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
