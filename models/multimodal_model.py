import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as transforms
from PIL import Image

class VisionEncoder(nn.Module):
    """
    Encoder de visão baseado em CNN + ViT
    Inspirado em arquiteturas do Hugging Face
    """
    def __init__(self, input_size=224, hidden_dim=768, num_classes=1000):
        super().__init__()
        
        # CNN para extração de características
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Projeção para dimensão do transformer
        self.projection = nn.Linear(512 * 7 * 7, hidden_dim)
        
        # Classificador final
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Extrair características visuais
        features = self.cnn_backbone(x)
        features = features.view(features.size(0), -1)
        
        # Projetar para dimensão do transformer
        projected = self.projection(features)
        
        # Classificação
        logits = self.classifier(projected)
        
        return projected, logits

class LanguageEncoder(nn.Module):
    """
    Encoder de linguagem baseado em Transformer
    Inspirado em modelos do Hugging Face
    """
    def __init__(self, vocab_size=30000, hidden_dim=768, num_layers=6, num_heads=12):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(512, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        embedded = self.embedding(input_ids)
        
        # Adicionar posição encoding
        seq_len = embedded.size(1)
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)
        embedded = embedded + pos_enc
        
        # Transformer
        if attention_mask is not None:
            # Converter para formato do PyTorch
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
        
        output = self.transformer(embedded, src_key_padding_mask=attention_mask)
        
        return output

class MultimodalFusion(nn.Module):
    """
    Fusão multimodal inspirada em BLIP/CLIP
    """
    def __init__(self, hidden_dim=768, fusion_dim=1024):
        super().__init__()
        
        # Camadas de projeção
        self.vision_proj = nn.Linear(hidden_dim, fusion_dim)
        self.text_proj = nn.Linear(hidden_dim, fusion_dim)
        
        # Fusão cross-modal
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Camadas de saída
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, hidden_dim)
        )
        
    def forward(self, vision_features, text_features):
        # Projetar características
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # Cross-attention
        attended_vision, _ = self.cross_attention(
            vision_proj, text_proj, text_proj
        )
        
        # Concatenar e fundir
        fused = torch.cat([attended_vision, text_proj], dim=-1)
        output = self.fusion_layer(fused)
        
        return output

class EducationalMultimodalModel(nn.Module):
    """
    Modelo multimodal completo para jogo educacional
    """
    def __init__(self, vocab_size=30000, num_classes=1000, hidden_dim=768):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(hidden_dim=hidden_dim, num_classes=num_classes)
        self.language_encoder = LanguageEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.multimodal_fusion = MultimodalFusion(hidden_dim=hidden_dim)
        
        # Head para geração de respostas
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Head para explicações educativas
        self.explanation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, images, questions, question_mask=None):
        # Encodar imagem
        vision_features, vision_logits = self.vision_encoder(images)
        
        # Encodar pergunta
        text_features = self.language_encoder(questions, question_mask)
        
        # Pooling das características de texto
        text_pooled = text_features.mean(dim=1)
        
        # Fusão multimodal
        fused_features = self.multimodal_fusion(
            vision_features.unsqueeze(1), 
            text_pooled.unsqueeze(1)
        )
        fused_features = fused_features.squeeze(1)
        
        # Gerar respostas e explicações
        answer_logits = self.answer_head(fused_features)
        explanation_logits = self.explanation_head(fused_features)
        
        return {
            'vision_logits': vision_logits,
            'answer_logits': answer_logits,
            'explanation_logits': explanation_logits,
            'fused_features': fused_features
        }

# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    batch_size = 4
    vocab_size = 30000
    num_classes = 1000
    
    # Criar modelo
    model = EducationalMultimodalModel(vocab_size=vocab_size, num_classes=num_classes)
    
    # Dados de exemplo
    images = torch.randn(batch_size, 3, 224, 224)
    questions = torch.randint(0, vocab_size, (batch_size, 20))
    question_mask = torch.ones(batch_size, 20)
    
    # Forward pass
    outputs = model(images, questions, question_mask)
    
    print("Modelo criado com sucesso!")
    print(f"Vision logits shape: {outputs['vision_logits'].shape}")
    print(f"Answer logits shape: {outputs['answer_logits'].shape}")
    print(f"Explanation logits shape: {outputs['explanation_logits'].shape}")

