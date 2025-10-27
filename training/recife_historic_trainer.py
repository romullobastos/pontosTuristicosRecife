"""
Sistema de Reconhecimento de Pontos Históricos do Recife
Treina um modelo CNN para identificar locais históricos específicos
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

class RecifeHistoricDataset(Dataset):
    """Dataset para pontos históricos do Recife"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Pontos históricos do Recife
        self.recife_locations = {
            'marco_zero': 0,
            'praia_boa_viagem': 1,
            'casa_da_cultura': 2,
            'forte_das_cinco_pontas': 3,
            'igreja_sao_pedro_dos_clerigos': 4,
            'rua_do_bom_jesus': 5,
            'ponte_mauricio_de_nassau': 6,
            'praça_do_arsenal': 7,
            'igreja_nossa_senhora_do_carmo': 8,
            'palacio_da_justica': 9,
            'teatro_santa_isabel': 10,
            'mercado_sao_jose': 11,
            'igreja_santo_antonio': 12,
            'rua_aurora': 13,
            'ponte_ponte_boa_viagem': 14,
            'parque_dona_lindu': 15,
            'museu_cidade_do_recife': 16,
            'igreja_madre_de_deus': 17,
            'rua_imperador_dom_pedro_ii': 18,
            'praça_rio_branco': 19
        }
        
        self.idx_to_class = {v: k for k, v in self.recife_locations.items()}
        self.class_to_idx = self.recife_locations
        
        # Carregar dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Carrega dataset de pontos históricos"""
        print("Carregando dataset de pontos historicos do Recife...")
        
        # Criar diretórios se não existirem
        for location in self.recife_locations.keys():
            location_dir = self.data_dir / location
            location_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar imagens existentes
        for location_dir in self.data_dir.iterdir():
            if location_dir.is_dir():
                location_name = location_dir.name.lower()
                if location_name in self.class_to_idx:
                    for img_file in location_dir.glob('*.jpg'):
                        self.images.append(str(img_file))
                        self.labels.append(self.class_to_idx[location_name])
        
        print(f"Dataset carregado: {len(self.images)} imagens, {len(self.class_to_idx)} locais historicos")
    
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

class RecifeHistoricCNN(nn.Module):
    """CNN otimizada para reconhecimento de pontos históricos"""
    
    def __init__(self, num_classes=20):
        super(RecifeHistoricCNN, self).__init__()
        
        # Encoder visual otimizado para arquitetura histórica
        self.features = nn.Sequential(
            # Primeira camada - detectar formas básicas
            nn.Conv2d(3, 32, kernel_size=7, padding=3),  # Kernel maior para arquitetura
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Segunda camada - detectar padrões arquitetônicos
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Terceira camada - detectar detalhes específicos
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Quarta camada - características complexas
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Quinta camada - características específicas do local
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classificador específico para pontos históricos
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class RecifeHistoricTrainer:
    """Treinador especializado em pontos históricos do Recife"""
    
    def __init__(self, data_dir='data/recife_historic'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Configurar transforms otimizados para arquitetura
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Maior resolução para detalhes arquitetônicos
            transforms.RandomHorizontalFlip(p=0.3),  # Flip horizontal para variação
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variação de cor
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Carregar dataset
        self.dataset = RecifeHistoricDataset(data_dir, self.transform)
        self.model = RecifeHistoricCNN(len(self.dataset.class_to_idx))
        self.model.to(self.device)
        
        # Configurar otimizador e loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)  # LR menor para fine-tuning
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Modelo criado: {sum(p.numel() for p in self.model.parameters())} parametros")
        print(f"Locais historicos suportados: {len(self.dataset.class_to_idx)}")
    
    def train(self, epochs=50, batch_size=16):
        """Treina o modelo para reconhecimento de pontos históricos"""
        if len(self.dataset) == 0:
            print("Nenhuma imagem encontrada para treinamento!")
            print("Adicione imagens em data/recife_historic/[nome_do_local]/")
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
    
    def save_model(self, model_path='models/recife_historic_model.pth'):
        """Salva o modelo treinado"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Salvar modelo com informações dos locais
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.dataset.class_to_idx,
            'idx_to_class': self.dataset.idx_to_class,
            'model_architecture': 'RecifeHistoricCNN',
            'recife_locations': self.dataset.recife_locations
        }, model_path)
        
        print(f"Modelo salvo em: {model_path}")
    
    def load_model(self, model_path='models/recife_historic_model.pth'):
        """Carrega modelo treinado"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.dataset.class_to_idx = checkpoint['class_to_idx']
            self.dataset.idx_to_class = checkpoint['idx_to_class']
            print(f"Modelo carregado de: {model_path}")
            return True
        return False
    
    def predict(self, image_path):
        """Faz predição em uma imagem de ponto histórico"""
        self.model.eval()
        
        # Carregar e processar imagem
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)
        
        # Converter predição para nome do local
        predicted_class = self.dataset.idx_to_class[predicted.item()]
        confidence_score = confidence[0][predicted.item()].item()
        
        return predicted_class, confidence_score
    
    def get_location_info(self, location_name):
        """Retorna informações sobre o ponto histórico"""
        location_info = {
            'marco_zero': {
                'nome': 'Marco Zero',
                'descricao': 'Praça principal do Recife Antigo, marco histórico da cidade',
                'historia': 'Local onde a cidade foi fundada em 1537',
                'caracteristicas': 'Praça ampla, calçada de pedra, vista para o mar'
            },
            'praia_boa_viagem': {
                'nome': 'Praia de Boa Viagem',
                'descricao': 'Principal praia urbana do Recife',
                'historia': 'Uma das praias mais famosas do Nordeste',
                'caracteristicas': 'Areia clara, coqueiros, calçadão'
            },
            'casa_da_cultura': {
                'nome': 'Casa da Cultura',
                'descricao': 'Antiga Casa de Detenção, hoje centro cultural',
                'historia': 'Construída em 1855, transformada em centro cultural em 1976',
                'caracteristicas': 'Arquitetura neoclássica, celas transformadas em lojas'
            },
            'forte_das_cinco_pontas': {
                'nome': 'Forte das Cinco Pontas',
                'descricao': 'Fortaleza histórica do século XVII',
                'historia': 'Construído pelos holandeses em 1630',
                'caracteristicas': 'Formato pentagonal, muralhas altas, museu'
            },
            'igreja_sao_pedro_dos_clerigos': {
                'nome': 'Igreja de São Pedro dos Clérigos',
                'descricao': 'Igreja barroca do século XVIII',
                'historia': 'Construída entre 1728 e 1782',
                'caracteristicas': 'Torre alta, fachada barroca, interior dourado'
            }
        }
        
        return location_info.get(location_name, {
            'nome': location_name.replace('_', ' ').title(),
            'descricao': 'Ponto histórico do Recife',
            'historia': 'Local com importância histórica para a cidade',
            'caracteristicas': 'Arquitetura e características únicas'
        })

def create_sample_recife_dataset():
    """Cria dataset de exemplo com imagens sintéticas dos pontos históricos"""
    print("Criando dataset de exemplo para pontos historicos do Recife...")
    
    data_dir = Path('data/recife_historic')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Cores características dos pontos históricos
    recife_colors = {
        'marco_zero': (139, 69, 19),        # Marrom (pedra)
        'praia_boa_viagem': (255, 228, 196), # Bege (areia)
        'casa_da_cultura': (105, 105, 105),  # Cinza (pedra)
        'forte_das_cinco_pontas': (160, 82, 45), # Marrom escuro
        'igreja_sao_pedro_dos_clerigos': (255, 215, 0), # Dourado
        'rua_do_bom_jesus': (139, 69, 19),   # Marrom
        'ponte_mauricio_de_nassau': (192, 192, 192), # Prata
        'praça_do_arsenal': (34, 139, 34),   # Verde
        'igreja_nossa_senhora_do_carmo': (255, 215, 0), # Dourado
        'palacio_da_justica': (105, 105, 105) # Cinza
    }
    
    for location, color in recife_colors.items():
        location_dir = data_dir / location
        location_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar 5 imagens sintéticas para cada local
        for i in range(5):
            # Criar imagem com cor característica do local
            img_array = np.full((256, 256, 3), color, dtype=np.uint8)
            
            # Adicionar variação aleatória
            noise = np.random.randint(-40, 40, (256, 256, 3), dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Salvar imagem
            img = Image.fromarray(img_array)
            img_path = location_dir / f"{location}_{i}.jpg"
            img.save(img_path)
    
    print(f"Dataset de exemplo criado: {len(recife_colors)} locais historicos")

if __name__ == "__main__":
    # Criar dataset de exemplo se não existir
    if not os.path.exists('data/recife_historic'):
        create_sample_recife_dataset()
    
    # Treinar modelo
    trainer = RecifeHistoricTrainer()
    
    # Tentar carregar modelo existente
    if not trainer.load_model():
        print("Modelo nao encontrado, iniciando treinamento...")
        model = trainer.train(epochs=30)
        if model:
            trainer.save_model()
    
    print("Sistema de reconhecimento de pontos historicos do Recife pronto!")

