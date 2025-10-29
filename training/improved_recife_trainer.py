import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision import models as tv_models
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
    
    def __init__(self, data_dir, transform=None, is_training=True, indices=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.augmentation = AdvancedImageTransforms()
        self._load_dataset()

        # Subconjunto opcional por índices (para split train/val)
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
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
        # Mostrar distribuição por classe
        counts = {cls: 0 for cls in self.class_to_idx.keys()}
        for label in self.labels:
            counts[self.idx_to_class[label]] += 1
        print("Distribuição por classe:")
        for cls, cnt in counts.items():
            print(f" - {cls}: {cnt}")
    
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
    
    def __init__(self, data_dir='data/recife_historic', use_transfer: bool = False):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Semente para reprodutibilidade
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # Configurar transforms
        self.transforms = AdvancedImageTransforms()
        self.train_transform = self.transforms.train_transforms
        self.val_transform = self.transforms.val_transforms
        
        # Carregar dataset
        self.dataset = ImprovedRecifeHistoricDataset(data_dir, self.train_transform, is_training=True)
        self.num_classes = len(self.dataset.class_to_idx)
        self.use_transfer = use_transfer
        
        # Criar modelo: ImprovedCNN (do zero) ou ResNet18 (Transfer Learning)
        if self.use_transfer:
            print("Usando Transfer Learning: ResNet18 pré-treinada (ImageNet)")
            resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            in_features = resnet.fc.in_features
            resnet.fc = nn.Linear(in_features, self.num_classes)
            self.model = resnet.to(self.device)
        else:
            print("Usando modelo ImprovedCNN treinado do zero")
            self.model = ImprovedCNN(self.num_classes).to(self.device)
        
        # Configurar otimizador com learning rate scheduler
        # Hiperparâmetros ajustados para dataset pequeno
        base_lr = 1e-4 if self.use_transfer else 5e-4
        wd = 5e-4 if not self.use_transfer else 1e-4
        # Se Transfer Learning, regra: LR menor para backbone, maior para a cabeça
        if self.use_transfer:
            params = [
                {"params": [p for n, p in self.model.named_parameters() if not n.startswith("fc.")], "lr": base_lr},
                {"params": self.model.fc.parameters(), "lr": base_lr * 5},
            ]
            self.optimizer = optim.AdamW(params, weight_decay=wd)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function com label smoothing leve (evita subtreino em dataset pequeno)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        print(f"Modelo melhorado criado: {sum(p.numel() for p in self.model.parameters())} parâmetros")
    
    def _create_dataloaders(self, batch_size=8, val_split=0.2):
        """Cria DataLoaders de treino e validação com split estratificado por classe."""
        if len(self.dataset) == 0:
            return None, None

        # Índices por classe para estratificação
        class_to_indices = {c: [] for c in self.dataset.class_to_idx.keys()}
        for i, label in enumerate(self.dataset.labels):
            class_to_indices[self.dataset.idx_to_class[label]].append(i)

        train_indices = []
        val_indices = []
        for cls, idxs in class_to_indices.items():
            if len(idxs) == 0:
                continue
            random.shuffle(idxs)
            split = max(1, int(len(idxs) * val_split))
            val_indices.extend(idxs[:split])
            train_indices.extend(idxs[split:])

        # Garantir que não fique vazio
        if len(train_indices) == 0:
            train_indices = val_indices
            val_indices = []

        train_dataset = ImprovedRecifeHistoricDataset(self.data_dir, self.train_transform, is_training=True, indices=train_indices)
        val_dataset = ImprovedRecifeHistoricDataset(self.data_dir, self.val_transform, is_training=False, indices=val_indices) if len(val_indices) > 0 else None

        # Sampler ponderado por classe para balancear treino
        if len(train_dataset.labels) > 0:
            label_counts = {}
            for lbl in train_dataset.labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            class_weights = {lbl: 1.0 / cnt for lbl, cnt in label_counts.items()}
            sample_weights = [class_weights[lbl] for lbl in train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )

        return train_loader, val_loader

    def train(self, epochs=30, batch_size=8):
        """Treinamento melhorado com técnicas avançadas"""
        if len(self.dataset) == 0:
            print("Nenhuma imagem encontrada para treinamento!")
            return None

        # Criar DataLoaders de treino e validação
        train_loader, val_loader = self._create_dataloaders(batch_size=batch_size, val_split=0.2)

        print(f"Iniciando treinamento melhorado: {epochs} épocas, batch_size={batch_size}")
        if val_loader is not None:
            print(f"Split: treino={len(train_loader.dataset)} | validação={len(val_loader.dataset)}")
        else:
            print(f"Dataset: {len(train_loader.dataset)} imagens (sem validação)")

        best_val_acc = -1
        best_model_state = None
        patience = 8
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = total_loss / max(1, len(train_loader))
            train_acc = 100.0 * correct / max(1, total)

            # Validação
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for v_images, v_labels in val_loader:
                        v_images, v_labels = v_images.to(self.device), v_labels.to(self.device)
                        v_outputs = self.model(v_images)
                        v_loss = self.criterion(v_outputs, v_labels)
                        val_loss += v_loss.item()
                        _, v_pred = torch.max(v_outputs, 1)
                        val_total += v_labels.size(0)
                        val_correct += (v_pred == v_labels).sum().item()

                val_loss = val_loss / max(1, len(val_loader))
                val_acc = 100.0 * val_correct / max(1, val_total)
                self.scheduler.step(val_loss)
            else:
                # Sem validação, use treino para scheduler (menos ideal)
                self.scheduler.step(train_loss)

            print_msg = f"Época {epoch+1}/{epochs}: TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.2f}%"
            if val_loader is not None:
                print_msg += f", ValLoss={val_loss:.4f}, ValAcc={val_acc:.2f}%"
            print_msg += f", LR={self.optimizer.param_groups[0]['lr']:.6f}"
            print(print_msg)

            # Early stopping baseado em ValAcc
            current_score = val_acc if val_loader is not None else train_acc
            if current_score > best_val_acc:
                best_val_acc = current_score
                best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping por estagnação na validação.")
                    break

            if train_acc > 99 and (val_loader is None or val_acc > 95):
                print("Convergência alcançada!")
                break

        # Carregar melhor modelo
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            tag = "ValAcc" if val_loader is not None else "TrainAcc"
            print(f"Melhor modelo carregado com {tag}: {best_val_acc:.2f}%")

        print("Treinamento melhorado concluído!")
        return self.model
    
    def save_model(self, model_path='models/improved_recife_historic_model.pth'):
        """Salva o modelo melhorado"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        arch_name = 'ResNet18' if getattr(self, 'use_transfer', False) else 'ImprovedCNN'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.dataset.class_to_idx,
            'idx_to_class': self.dataset.idx_to_class,
            'model_architecture': arch_name,
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
            arch_name = checkpoint.get('model_architecture', 'ImprovedCNN')
            # Heurística: se metadado estiver errado/ausente, detectar pelo formato das chaves
            state_keys = list(checkpoint['model_state_dict'].keys())
            looks_like_resnet = any(k.startswith('layer1.') or k.startswith('conv1') for k in state_keys)
            if arch_name == 'ResNet18' or looks_like_resnet:
                # Recriar ResNet18 com a cabeça certa e carregar pesos
                resnet = tv_models.resnet18(weights=None)
                in_features = resnet.fc.in_features
                resnet.fc = nn.Linear(in_features, self.num_classes)
                self.model = resnet.to(self.device)
                self.use_transfer = True
            else:
                self.model = ImprovedCNN(self.num_classes).to(self.device)
                self.use_transfer = False
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
