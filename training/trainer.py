import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

class EducationalDataset(Dataset):
    """
    Dataset para treinamento do modelo multimodal educacional
    """
    
    def __init__(self, data_path, transform=None, tokenizer=None, max_length=50):
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Carregar dados
        self.data = self._load_data()
        
    def _load_data(self):
        """Carrega os dados do dataset"""
        # Estrutura esperada:
        # data/
        #   ├── recife_historic/
        #   └── annotations.json
        
        annotations_path = os.path.join(self.data_path, "annotations.json")
        
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Criar dataset de exemplo se não existir
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self):
        """Cria um dataset de exemplo para demonstração"""
        sample_data = [
            {
                "image_path": "data/recife_historic/marco_zero/marco-zero-6-1030x579.jpeg",
                "question": "Que local histórico é este?",
                "answer": "Este é o Marco Zero do Recife",
                "explanation": "O Marco Zero é a praça principal do Recife Antigo, onde a cidade foi fundada em 1537",
                "category": "historic_location",
                "difficulty": "medium",
                "points": 20
            },
            {
                "image_path": "data/recife_historic/casa_da_cultura/bdf-20250516-070544-8be936-750x400.jpeg", 
                "question": "Que local histórico é este?",
                "answer": "Esta é a Casa da Cultura",
                "explanation": "A Casa da Cultura é um importante centro cultural do Recife, localizado no antigo prédio da Casa de Detenção",
                "category": "historic_location",
                "difficulty": "medium",
                "points": 20
            },
            {
                "image_path": "data/recife_historic/forte_das_cinco_pontas/museu-do-recife-forte-das-cinco-pontas-foto-arthur-de-souza-31.jpg",
                "question": "Que local histórico é este?",
                "answer": "Este é o Forte das Cinco Pontas",
                "explanation": "O Forte das Cinco Pontas é uma fortaleza histórica construída pelos holandeses em 1630",
                "category": "historic_location",
                "difficulty": "hard",
                "points": 25
            }
        ]
        
        # Salvar dataset de exemplo
        os.makedirs(self.data_path, exist_ok=True)
        with open(os.path.join(self.data_path, "annotations.json"), 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        return sample_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Carregar imagem
        image_path = os.path.join(self.data_path, item["image_path"])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Se a imagem não existir, criar uma imagem aleatória
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        # Tokenizar texto
        if self.tokenizer:
            question_tokens = self.tokenizer.encode(
                item["question"], 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            )
            answer_tokens = self.tokenizer.encode(
                item["answer"], 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            )
        else:
            # Tokenização simples se não houver tokenizer
            question_tokens = [hash(word) % 1000 for word in item["question"].split()[:self.max_length]]
            answer_tokens = [hash(word) % 1000 for word in item["answer"].split()[:self.max_length]]
            
            # Padding
            question_tokens = question_tokens + [0] * (self.max_length - len(question_tokens))
            answer_tokens = answer_tokens + [0] * (self.max_length - len(answer_tokens))
        
        return {
            'image': image,
            'question': torch.tensor(question_tokens, dtype=torch.long),
            'answer': torch.tensor(answer_tokens, dtype=torch.long),
            'category': item["category"],
            'difficulty': item["difficulty"],
            'points': item["points"]
        }

class SimpleTokenizer:
    """Tokenizador simples para demonstração"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Constrói vocabulário básico"""
        # Palavras comuns em português
        common_words = [
            "que", "é", "este", "esta", "um", "uma", "de", "da", "do", "das", "dos",
            "recife", "histórico", "local", "praça", "igreja", "forte", "teatro", "marco",
            "casa", "rua", "avenida", "ponte", "praia", "cultura", "centro", "história",
            "barroca", "colonial", "neoclássica", "arquitetura", "monumento", "turismo", "cidade"
        ]
        
        for i, word in enumerate(common_words):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
    
    def encode(self, text, max_length=50, padding='max_length', truncation=True):
        """Codifica texto para tokens"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(1)  # Token desconhecido
        
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        if padding == 'max_length':
            tokens = tokens + [0] * (max_length - len(tokens))
        
        return tokens

class TrainingPipeline:
    """
    Pipeline de treinamento para o modelo multimodal educacional
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Métricas de treinamento
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(dataloader, desc="Treinando")
        
        for batch in progress_bar:
            # Mover dados para device
            images = batch['image'].to(self.device)
            questions = batch['question'].to(self.device)
            answers = batch['answer'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images, questions)
            
            # Calcular loss
            vision_loss = criterion(outputs['vision_logits'], answers)
            answer_loss = criterion(outputs['answer_logits'], answers)
            explanation_loss = criterion(outputs['explanation_logits'], answers)
            
            total_loss_batch = vision_loss + answer_loss + explanation_loss
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Calcular acurácia (simplificada)
            _, predicted = torch.max(outputs['vision_logits'], 1)
            _, target = torch.max(answers, 1)
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.size(0)
            
            # Atualizar progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader, criterion):
        """Valida uma época"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validando"):
                # Mover dados para device
                images = batch['image'].to(self.device)
                questions = batch['question'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, questions)
                
                # Calcular loss
                vision_loss = criterion(outputs['vision_logits'], answers)
                answer_loss = criterion(outputs['answer_logits'], answers)
                explanation_loss = criterion(outputs['explanation_logits'], answers)
                
                total_loss_batch = vision_loss + answer_loss + explanation_loss
                total_loss += total_loss_batch.item()
                
                # Calcular acurácia
                _, predicted = torch.max(outputs['vision_logits'], 1)
                _, target = torch.max(answers, 1)
                correct_predictions += (predicted == target).sum().item()
                total_predictions += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_dataloader, val_dataloader, epochs=10, lr=0.001):
        """Treina o modelo"""
        # Otimizador e loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Scheduler para learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        print(f"Iniciando treinamento por {epochs} épocas...")
        print(f"Device: {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nÉpoca {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Treinar
            train_loss, train_acc = self.train_epoch(train_dataloader, optimizer, criterion)
            
            # Validar
            val_loss, val_acc = self.validate_epoch(val_dataloader, criterion)
            
            # Atualizar scheduler
            scheduler.step(val_loss)
            
            # Salvar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print resultados
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Novo melhor modelo salvo!")
        
        print("\nTreinamento concluído!")
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plota as curvas de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Importar modelo
    import sys
    sys.path.append('..')
    from models.multimodal_model import EducationalMultimodalModel
    
    # Configurações
    batch_size = 8
    epochs = 5
    learning_rate = 0.001
    
    # Criar tokenizer
    tokenizer = SimpleTokenizer()
    
    # Transformações para imagens
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Criar dataset
    dataset = EducationalDataset(
        data_path='../data',
        transform=transform,
        tokenizer=tokenizer
    )
    
    # Dividir em train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Criar dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Criar modelo
    model = EducationalMultimodalModel(vocab_size=tokenizer.vocab_size)
    
    # Criar pipeline de treinamento
    trainer = TrainingPipeline(model)
    
    # Treinar
    trainer.train(train_loader, val_loader, epochs=epochs, lr=learning_rate)

