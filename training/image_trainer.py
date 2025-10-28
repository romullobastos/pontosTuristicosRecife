"""
Sistema de Treinamento Real para Reconhecimento de Imagens
Treina um modelo CNN do zero para identificar locais históricos do Recife
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
from pathlib import Path

class ImageDataset(Dataset):
    """Dataset para treinamento com imagens reais"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Carregar imagens e labels
        self._load_dataset()
    
    def _load_dataset(self):
        """Carrega dataset de imagens"""
        print("Carregando dataset de imagens...")
        
        # Criar diretórios se não existirem
        categories = ['recife_historic']
        for category in categories:
            category_dir = self.data_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapear classes - locais históricos do Recife
        self.class_to_idx = {
            'marco_zero': 0, 'casa_da_cultura': 1, 'forte_das_cinco_pontas': 2,
            'igreja_madre_de_deus': 3, 'igreja_nossa_senhora_do_carmo': 4,
            'igreja_santo_antonio': 5, 'igreja_sao_pedro_dos_clerigos': 6,
            'marco_zero': 7, 'mercado_sao_jose': 8, 'palacio_da_justica': 9,
            'rua_aurora': 10, 'rua_do_bom_jesus': 11, 'teatro_santa_isabel': 12
        }
        
        # Carregar imagens existentes
        for category_dir in self.data_dir.iterdir():
            if category_dir.is_dir():
                # Usar o nome da pasta como classe
                class_name = category_dir.name.lower()
                if class_name in self.class_to_idx:
                    for img_file in category_dir.glob('*.jpg'):
                        self.images.append(str(img_file))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Dataset carregado: {len(self.images)} imagens, {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Carregar imagem
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleCNN(nn.Module):
    """CNN simples para classificação de imagens"""
    
    def __init__(self, num_classes=29):
        super(SimpleCNN, self).__init__()
        
        # Encoder visual (CNN)
        self.features = nn.Sequential(
            # Primeira camada convolucional
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Segunda camada convolucional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Terceira camada convolucional
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Quarta camada convolucional
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class ImageTrainer:
    """Treinador para modelo de reconhecimento de imagens"""
    
    def __init__(self, data_dir='data/training_images'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Configurar transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Carregar dataset
        self.dataset = ImageDataset(data_dir, self.transform)
        self.model = SimpleCNN(len(self.dataset.class_to_idx))
        self.model.to(self.device)
        
        # Configurar otimizador e loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Modelo criado: {sum(p.numel() for p in self.model.parameters())} parametros")
    
    def train(self, epochs=50, batch_size=32):
        """Treina o modelo"""
        if len(self.dataset) == 0:
            print("Nenhuma imagem encontrada para treinamento!")
            print("Adicione imagens em data/training_images/")
            return None
        
        # Criar DataLoader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Iniciando treinamento: {epochs} epocas, batch_size={batch_size}")
        print(f"Dataset: {len(self.dataset)} imagens")
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Estatísticas
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calcular métricas da época
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            print(f"Epoca {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        print("Treinamento concluido!")
        return self.model
    
    def save_model(self, model_path='models/trained_image_model.pth'):
        """Salva o modelo treinado"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Salvar modelo
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.dataset.class_to_idx,
            'model_architecture': 'SimpleCNN'
        }, model_path)
        
        print(f"Modelo salvo em: {model_path}")
    
    def load_model(self, model_path='models/trained_image_model.pth'):
        """Carrega modelo treinado"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.dataset.class_to_idx = checkpoint['class_to_idx']
            print(f"Modelo carregado de: {model_path}")
            return True
        return False
    
    def predict(self, image_path):
        """Faz predição em uma imagem"""
        self.model.eval()
        
        # Carregar e processar imagem
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)
        
        # Converter predição para nome da classe
        idx_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}
        predicted_class = idx_to_class[predicted.item()]
        confidence_score = confidence[0][predicted.item()].item()
        
        return predicted_class, confidence_score

def create_sample_dataset():
    """Cria dataset de exemplo com imagens sintéticas"""
    print("Criando dataset de exemplo...")
    
    data_dir = Path('data/training_images')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar imagens sintéticas para cada classe - locais históricos do Recife
    classes = {
        'marco_zero': (139, 69, 19),           # Marrom (calçada portuguesa)
        'casa_da_cultura': (105, 105, 105),     # Cinza (arquitetura antiga)
        'forte_das_cinco_pontas': (160, 82, 45),# Marrom escuro (pedra)
        'teatro_santa_isabel': (192, 192, 192), # Branco/Prata (neoclássico)
        'igreja_madre_de_deus': (255, 250, 240), # Bege (barroca)
        'igreja_nossa_senhora_do_carmo': (139, 0, 0), # Vermelho escuro
        'palacio_da_justica': (245, 245, 220), # Branco/Bege (institucional)
        'rua_do_bom_jesus': (218, 165, 32),    # Dourado (casas coloridas)
    }
    
    for class_name, color in classes.items():
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar 10 imagens sintéticas para cada classe
        for i in range(10):
            # Criar imagem com cor característica da classe
            img_array = np.full((224, 224, 3), color, dtype=np.uint8)
            
            # Adicionar variação aleatória
            noise = np.random.randint(-30, 30, (224, 224, 3), dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Salvar imagem
            img = Image.fromarray(img_array)
            img_path = class_dir / f"{class_name}_{i}.jpg"
            img.save(img_path)
    
    print(f"Dataset de exemplo criado: {len(classes)} classes")

if __name__ == "__main__":
    # Criar dataset de exemplo se não existir
    if not os.path.exists('data/training_images'):
        create_sample_dataset()
    
    # Treinar modelo
    trainer = ImageTrainer()
    
    # Tentar carregar modelo existente
    if not trainer.load_model():
        print("Modelo nao encontrado, iniciando treinamento...")
        model = trainer.train(epochs=20)
        if model:
            trainer.save_model()
    
    print("Sistema de treinamento pronto!")
