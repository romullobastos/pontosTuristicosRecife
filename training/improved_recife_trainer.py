import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import os
from pathlib import Path
import numpy as np
import json
import random

# Data Augmentation Avançado
class AdvancedImageTransforms:
    """Transformações avançadas para melhorar distinção de imagens"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # Transformações básicas
        self.basic_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transformações de treinamento com augmentation MELHORADO
        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Mais flip para diferentes ângulos
            transforms.RandomRotation(10),  # Rotação para simular diferentes ângulos
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translação
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transformações de validação
        self.val_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def apply_advanced_augmentation(self, image):
        """Aplica augmentation avançado personalizado"""
        # Converter para PIL se necessário
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Aplicar filtros específicos para pontos históricos
        if random.random() < 0.5:
            # Filtro de nitidez para destacar detalhes arquitetônicos
            image = image.filter(ImageFilter.SHARPEN)
        
        if random.random() < 0.4:
            # Ajuste de contraste para melhorar definição
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
        
        if random.random() < 0.4:
            # Ajuste de brilho
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < 0.3:
            # Ajuste de saturação (útil para distinguir cores arquitetônicas)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        return image

# Arquitetura CNN Melhorada (sem BatchNorm para funcionar com datasets pequenos)
class ImprovedCNN(nn.Module):
    """CNN melhorada com técnicas avançadas para distinção de pontos históricos"""
    
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()
        
        # Feature extractor sem BatchNorm para funcionar com datasets pequenos
        self.features = nn.Sequential(
            # Primeira camada - características básicas
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Segunda camada - características intermediárias
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Terceira camada - características complexas
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Quarta camada - características específicas
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classificador sem BatchNorm
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            
            nn.Linear(256, num_classes)
        )
        
        # Inicialização de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos com Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

# Dataset Melhorado
class ImprovedRecifeHistoricDataset(Dataset):
    """Dataset melhorado com augmentation e pré-processamento"""
    
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.augmentation = AdvancedImageTransforms()
        self._load_dataset()
    
    def _load_dataset(self):
        """Carrega dataset com informações detalhadas"""
        print("Carregando dataset melhorado de pontos históricos do Recife...")
        
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        idx_counter = 0
        
        # Carregar imagens existentes
        for category_dir in self.data_dir.iterdir():
            if category_dir.is_dir():
                class_name = category_dir.name.lower()
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = idx_counter
                    self.idx_to_class[idx_counter] = class_name
                    idx_counter += 1
                
                # Carregar arquivos jpg, jpeg e png
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_file in category_dir.glob(ext):
                        self.images.append(str(img_file))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Dataset carregado: {len(self.images)} imagens, {len(self.class_to_idx)} locais históricos")
        print(f"Locais históricos suportados: {len(self.idx_to_class)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Aplicar augmentation se for treinamento (reduzido)
        if self.is_training and random.random() < 0.3:
            image = self.augmentation.apply_advanced_augmentation(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Treinador Melhorado
class ImprovedRecifeHistoricTrainer:
    """Treinador melhorado com técnicas avançadas"""
    
    def __init__(self, data_dir='data/recife_historic'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Configurar transforms
        self.transforms = AdvancedImageTransforms()
        self.train_transform = self.transforms.train_transforms
        self.val_transform = self.transforms.val_transforms
        
        # Carregar dataset
        self.dataset = ImprovedRecifeHistoricDataset(data_dir, self.train_transform, is_training=True)
        self.num_classes = len(self.dataset.class_to_idx)
        
        # Criar modelo melhorado
        self.model = ImprovedCNN(self.num_classes).to(self.device)
        
        # Configurar otimizador com learning rate scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function com label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"Modelo melhorado criado: {sum(p.numel() for p in self.model.parameters())} parâmetros")
    
    def train(self, epochs=30, batch_size=8):
        """Treinamento melhorado com técnicas avançadas"""
        if len(self.dataset) == 0:
            print("Nenhuma imagem encontrada para treinamento!")
            return None
        
        # Criar DataLoader com workers
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # 0 para Windows
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Iniciando treinamento melhorado: {epochs} épocas, batch_size={batch_size}")
        print(f"Dataset: {len(self.dataset)} imagens")
        
        best_accuracy = 0
        best_model_state = None
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping para estabilidade
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calcular métricas da época
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            # Atualizar learning rate
            self.scheduler.step(avg_loss)
            
            # Salvar melhor modelo
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = self.model.state_dict().copy()
            
            print(f"Época {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, LR={self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping se accuracy muito baixa (removido para permitir treinamento completo)
            # if epoch > 10 and accuracy < 20:
            #     print("Early stopping devido à baixa accuracy")
            #     break
            
            # Early stopping se accuracy muito alta (convergência)
            if accuracy > 95:
                print("Convergência alcançada!")
                break
        
        # Carregar melhor modelo
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Melhor modelo carregado com accuracy: {best_accuracy:.2f}%")
        
        print("Treinamento melhorado concluído!")
        return self.model
    
    def save_model(self, model_path='models/improved_recife_historic_model.pth'):
        """Salva o modelo melhorado"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.dataset.class_to_idx,
            'idx_to_class': self.dataset.idx_to_class,
            'model_architecture': 'ImprovedCNN',
            'num_classes': self.num_classes
        }, model_path)
        
        print(f"Modelo melhorado salvo em: {model_path}")
    
    def load_model(self, model_path='models/improved_recife_historic_model.pth'):
        """Carrega modelo melhorado"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.dataset.class_to_idx = checkpoint['class_to_idx']
            self.dataset.idx_to_class = checkpoint['idx_to_class']
            self.num_classes = checkpoint['num_classes']
            self.model = ImprovedCNN(self.num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo melhorado carregado de: {model_path}")
            return True
        return False
    
    def predict(self, image_path):
        """Predição melhorada com técnicas avançadas"""
        self.model.eval()
        
        # Carregar e processar imagem
        image = Image.open(image_path).convert('RGB')
        
        # Aplicar transformações de validação
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Predição com confidence
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Converter predição para nome do local
        predicted_class = self.dataset.idx_to_class[predicted.item()]
        confidence_score = confidence[0].item()
        
        return predicted_class, confidence_score
    
    def get_location_info(self, location_name):
        """Retorna informações detalhadas sobre pontos históricos"""
        location_info = {
            'casa_da_cultura': {
                'nome': 'Casa da Cultura',
                'descricao': 'Antiga Casa de Detenção, hoje centro cultural',
                'historia': 'Construída em 1855, transformada em centro cultural em 1976',
                'caracteristicas': 'Arquitetura neoclássica, celas transformadas em lojas'
            },
            'forte_das_cinco_pontas': {
                'nome': 'Forte das Cinco Pontas',
                'descricao': 'Fortaleza histórica construída pelos holandeses',
                'historia': 'Construído em 1630, formato pentagonal único',
                'caracteristicas': 'Arquitetura militar colonial, formato pentagonal'
            },
            'igreja_madre_de_deus': {
                'nome': 'Igreja Madre de Deus',
                'descricao': 'Igreja histórica no bairro do Recife',
                'historia': 'Século XVIII, arquitetura barroca',
                'caracteristicas': 'Fachada barroca, torre alta'
            },
            'igreja_nossa_senhora_do_carmo': {
                'nome': 'Igreja Nossa Senhora do Carmo',
                'descricao': 'Igreja histórica com arquitetura barroca',
                'historia': 'Importante marco religioso',
                'caracteristicas': 'Interior ricamente decorado'
            },
            'igreja_santo_antonio': {
                'nome': 'Igreja Santo Antônio',
                'descricao': 'Igreja histórica do Recife',
                'historia': 'Século XVIII',
                'caracteristicas': 'Arquitetura colonial barroca'
            },
            'igreja_sao_pedro_dos_clerigos': {
                'nome': 'Igreja de São Pedro dos Clérigos',
                'descricao': 'Igreja barroca do século XVIII',
                'historia': 'Construída entre 1728 e 1782',
                'caracteristicas': 'Torre alta, fachada barroca elaborada'
            },
            'marco_zero': {
                'nome': 'Marco Zero',
                'descricao': 'Praça principal do Recife Antigo, marco histórico da cidade',
                'historia': 'Local onde a cidade foi fundada em 1537',
                'caracteristicas': 'Praça ampla, calçada de pedra, vista para o mar'
            },
            'mercado_sao_jose': {
                'nome': 'Mercado São José',
                'descricao': 'Mercado público histórico',
                'historia': 'Construído em 1875',
                'caracteristicas': 'Arquitetura de ferro pré-fabricada'
            },
            'palacio_da_justica': {
                'nome': 'Palácio da Justiça',
                'descricao': 'Tribunal de Justiça de Pernambuco',
                'historia': 'Prédio histórico com arquitetura neoclássica',
                'caracteristicas': 'Fachada imponente, colunas clássicas'
            },
            'rua_aurora': {
                'nome': 'Rua Aurora',
                'descricao': 'Rua histórica do centro do Recife',
                'historia': 'Uma das principais ruas do Recife',
                'caracteristicas': 'Arquitetura eclética, prédios históricos'
            },
            'rua_do_bom_jesus': {
                'nome': 'Rua do Bom Jesus',
                'descricao': 'Uma das ruas mais antigas do Recife',
                'historia': 'Localizada no Recife Antigo',
                'caracteristicas': 'Casas coloridas, arquitetura colonial'
            },
            'teatro_santa_isabel': {
                'nome': 'Teatro Santa Isabel',
                'descricao': 'Teatro histórico do Recife',
                'historia': 'Construído em 1850, neoclássico',
                'caracteristicas': 'Arquitetura neoclássica, fachada imponente'
            }
        }
        
        return location_info.get(location_name, {
            'nome': location_name.replace('_', ' ').title(),
            'descricao': 'Ponto histórico do Recife',
            'historia': 'Local com importância histórica para a cidade',
            'caracteristicas': 'Arquitetura e características únicas'
        })
    
    def compare_images(self, img1, img2):
        """
        Compara duas imagens retornando similaridade (0-1)
        Método: Classifica ambas as imagens e verifica se são do MESMO local histórico
        """
        self.model.eval()
        
        try:
            # Processar ambas as imagens
            img1_tensor = self.val_transform(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.val_transform(img2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Passar pelo modelo para obter predições
                output1 = self.model(img1_tensor)
                output2 = self.model(img2_tensor)
                
                # Obter probabilidades
                probs1 = torch.nn.functional.softmax(output1, dim=1)
                probs2 = torch.nn.functional.softmax(output2, dim=1)
                
                # Pegar as classes preditas
                _, pred_class1 = torch.max(output1, 1)
                _, pred_class2 = torch.max(output2, 1)
                
                # Pegar as probabilidades das classes preditas
                confidence1 = probs1[0][pred_class1].item()
                confidence2 = probs2[0][pred_class2].item()
                
                # Verificar se são da MESMA classe (mesmo local histórico)
                same_location = (pred_class1 == pred_class2).item()
                
                if same_location:
                    # São do mesmo local! Calcular similaridade baseada na média das confianças
                    similarity = (confidence1 + confidence2) / 2
                    
                    # Se as confianças são altas, ambas estão bem classificadas
                    if confidence1 > 0.7 and confidence2 > 0.7:
                        similarity = min(0.95, similarity * 1.1)  # Boost para fotos bem classificadas
                else:
                    # São de locais diferentes! 
                    # Calcular dissimilaridade baseada na diferença entre as predições
                    diff = abs(confidence1 - confidence2)
                    
                    # Se uma foto tem confiança alta e a outra é claramente diferente
                    if confidence1 > 0.8 or confidence2 > 0.8:
                        similarity = 0.1  # Muito baixa similaridade
                    else:
                        # Ambas mal classificadas ou incertas
                        similarity = 0.3  # Similaridade baixa
                
                # Garantir que está entre 0.1 e 0.95
                similarity = max(0.1, min(0.95, similarity))
                
                return similarity
                
        except Exception as e:
            print(f"Erro ao comparar imagens: {e}")
            import traceback
            traceback.print_exc()
            # Retornar uma similaridade mínima em caso de erro
            return 0.1

if __name__ == "__main__":
    # Testar o sistema melhorado
    trainer = ImprovedRecifeHistoricTrainer()
    
    # Tentar carregar modelo existente
    if not trainer.load_model():
        print("Modelo não encontrado, iniciando treinamento melhorado...")
        model = trainer.train(epochs=30, batch_size=8)
        if model:
            trainer.save_model()
    
    print("Sistema melhorado pronto!")
