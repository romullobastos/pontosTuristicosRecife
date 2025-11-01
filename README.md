# 🏛️ Aplicativo Educacional - Pontos Históricos do Recife

Sistema educacional gamificado que utiliza **Deep Learning** para reconhecimento de pontos históricos do Recife através de imagens.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Tecnologias e Modelos](#tecnologias-e-modelos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Manutenção](#manutenção)
- [Relatório Técnico](#relatório-técnico)
- [Contribuição](#contribuiçãocontribuição)

> 📖 **Para pessoas sem conhecimento técnico:** Consulte o [GUIA_LEIGO.md](GUIA_LEIGO.md) que explica o projeto de forma simples e acessível!

## 🎮 Como Jogar

1. **Modo Identificação**: 
   - Envie uma foto de um ponto turístico do Recife
   - O sistema identificará o local e fornecerá informações históricas
   
2. **Modo Mistério**:
   - Tente adivinhar qual é o local histórico mostrado na imagem
   - Use as dicas disponíveis para ajudar na identificação
   - Ganhe pontos por acertos consecutivos

3. **Sistema de Pontuação**:
   - Acertos consecutivos aumentam sua pontuação
   - Dicas utilizadas reduzem a pontuação final
   - Desafios diários com recompensas especiais

## 🎯 Sobre o Projeto

Este projeto implementa um **sistema de reconhecimento de imagens** especializado em identificar pontos históricos do Recife. Utiliza **Redes Neurais Convolucionais (CNNs)** treinadas especificamente com fotografias reais de locais históricos da cidade.

### Funcionalidades Principais

- 📸 **Reconhecimento de Imagens**: Identifica pontos históricos do Recife
- 🎮 **Sistema de Gamificação**: Pontuação, níveis e conquistas
- 📚 **Educacional**: Explicações históricas sobre cada local
- ⚡ **Tempo Real**: Respostas instantâneas
- 🏆 **Ranking**: Competição entre jogadores

## 🏗️ Tecnologias e Modelos

### Stack Tecnológico

- **PyTorch** - Framework de Deep Learning
- **Flask** - Servidor web
- **scikit-learn** e **NLTK** - NLP (TF-IDF, stopwords, lematização)
- **Python** - Linguagem principal
- **HTML/CSS/JavaScript** - Interface frontend

### Dependências Principais

O projeto utiliza apenas as dependências essenciais:

- `torch` e `torchvision` - Deep Learning
- `transformers` - Modelos de linguagem (multimodal)
- `flask` e `flask-cors` - Servidor web
- `scikit-learn` e `nltk` - Processamento de linguagem natural
- `matplotlib` e `seaborn` - Visualização (treinamento)
- `Pillow` e `numpy` - Processamento de imagens
- `python-docx` - Exportação de documentos

### Modelo de Deep Learning

**Arquitetura CNN (Convolutional Neural Network)**

```

Para comparação visual entre duas imagens (Modo Foto), a aplicação extrai **embeddings** do backbone (ResNet18 ou ImprovedCNN) e calcula a **similaridade de cosseno** entre os vetores, com pequenos ajustes de escala e reforço quando as classes coincidem com alta confiança.
ImprovedCNN:
├── Feature Extractor (Camadas Convolucionais)
│   ├── Conv2d(3→64) + ReLU
│   ├── Conv2d(64→128) + ReLU  
│   ├── Conv2d(128→256) + ReLU
│   ├── Conv2d(256→512) + ReLU
│   └── AdaptiveAvgPool2d(4×4)
├── Classifier (Dense Layers)
│   ├── Linear(8192→1024) + ReLU + Dropout(0.3)
│   ├── Linear(1024→512) + ReLU + Dropout(0.15)
│   ├── Linear(512→256) + ReLU + Dropout(0.09)
│   └── Linear(256→12)  # 12 locais históricos
└── Saída: Predição + Confiança
```

**Características Técnicas:**
- **Parâmetros**: 13.7 milhões
- **Classes**: 12 locais históricos
- **Dataset**: 60 imagens de alta qualidade distribuídas em 12 pastas de classes em `data/recife_historic/`. O arquivo `data/photo_descriptions.json` contém 60 entradas, cada uma com metadados completos, incluindo descrições, dicas e informações históricas.
- **Acurácia (conjunto experimental de 25 imagens)**: 96%  
  Nota: no retreinamento recente com Transfer Learning (ResNet18) obteve-se **ValAcc 100%** no split de validação (12 imagens).
- **Tempo de Treinamento**: 2-3 minutos
- **Batch Size**: 2 imagens/época
- **Learning Rate**: 0.001 (com scheduler)

**Atualização Recente (01/11/2024)**: 
- Corrigidos os caminhos das imagens no `photo_descriptions.json` para garantir carregamento consistente
- Padronização dos caminhos para iniciar com `data/`
- Verificação de integridade de todas as 60 imagens do dataset

Nota: modelo atual treinado via **Transfer Learning (ResNet18)** com 60 imagens (12 classes) obteve **ValAcc de 100%** no split de validação (12 imagens). A comparação visual entre fotos usa **embeddings** do backbone com **similaridade do cosseno**.

### 🏛️ Locais Históricos Reconhecidos

O sistema agora inclui um total de **60 imagens** distribuídas entre os seguintes pontos turísticos:

O modelo identifica **12 pontos históricos do Recife**:

1. 🏛️ **Casa da Cultura** - Antiga Casa de Detenção
2. 🏰 **Forte das Cinco Pontas** - Fortaleza holandesa
3. ⛪ **Igreja Madre de Deus** - Igreja histórica
4. ⛪ **Igreja Nossa Senhora do Carmo** - Arquitetura barroca
5. ⛪ **Igreja Santo Antônio** - Século XVIII
6. ⛪ **Igreja São Pedro dos Clérigos** - Torre alta barroca
7. 📍 **Marco Zero** - Praça principal do Recife Antigo
8. 🛒 **Mercado São José** - Mercado público histórico
9. ⚖️ **Palácio da Justiça** - Tribunal de PE
10. 🛣️ **Rua Aurora** - Rua histórica do centro
11. 🛣️ **Rua do Bom Jesus** - Uma das mais antigas do Recife
12. 🎭 **Teatro Santa Isabel** - Teatro neoclássico

## 🛠️ Estrutura do Projeto

```
appDeepLearning/
├── data/
│   ├── recife_historic/      # Imagens dos pontos turísticos
│   │   ├── casa_da_cultura/
│   │   ├── forte_das_cinco_pontas/
│   │   └── ...
│   ├── photo_descriptions.json  # Metadados das imagens
│   ├── players.json         # Dados dos jogadores
│   └── users.json           # Dados de usuários
├── game/
│   ├── gamification.py      # Lógica de gamificação
│   └── photo_description_game.py  # Jogo de descrição de fotos
├── templates/               # Templates HTML
├── main.py                 # Aplicação principal
└── README.md               # Documentação
```

## 🚀 Instalação

### Requisitos

- Python 3.8+
- 4GB RAM mínimo
- Espaço em disco: 500MB

### 📥 Instalação Passo a Passo

```bash
# 1. Clone o repositório
git clone [URL_DO_REPOSITORIO]
cd appDeepLearning

# 2. Crie e ative um ambiente virtual (recomendado)
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe recursos do NLTK (necessário apenas na primeira execução)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 5. (Opcional) Verifique se as fotos estão organizadas
# Estrutura esperada:
# data/recife_historic/
#   ├── marco_zero/
#   ├── casa_da_cultura/
#   ├── ... (outros locais)
```

## 🎓 Treinamento do Modelo

### Treinamento Automático (Recomendado)

Para treinar ou retreinar o modelo com suas fotos:

```bash
python RETREINAR_MODELO.py
```

### Como Adicionar Novas Fotos

1. **Adicione as fotos** em `data/recife_historic/[nome_do_local]/`
   - Formatos aceitos: `.jpg`, `.jpeg`, `.png`
   - Exemplo: `data/recife_historic/marco_zero/nova_foto.jpg`

2. **Execute o retreinamento**:
   ```bash
   python RETREINAR_MODELO.py
   ```

3. **Aguarde o treinamento** (2-3 minutos)

4. **Teste o aplicativo**: http://localhost:5000

### Configuração de Treinamento

O treinamento usa:
- **Data Augmentation**: Rotação, brilho, contraste
- **Early Stopping**: Para automaticamente em 96% acurácia
- **Learning Rate Scheduling**: Reduz LR quando converge
- **Dropout**: 30% para regularização

## 🎮 Como Usar

### 1. Iniciar o Aplicativo

```bash
# Método 1: Script de inicialização
python start_game.py

# Método 2: Direto
python main.py

# Método 3: Via Flask
flask run
```

### 2. Acessar o Jogo

Abra o navegador em: **http://localhost:5000**

### 3. Como Jogar

1. **Criar Jogador**: Digite seu nome
2. **Enviar Foto**: Upload de imagem de um ponto histórico
3. **Fazer Pergunta**: Ex.: "Que local histórico é este?"
4. **Ver Resposta**: O sistema identifica o local e dá informações históricas
5. **Ganhar Pontos**: Pontuação baseada em similaridade/score (modos Foto/Descrição)

## 🔧 Manutenção

### Limpeza de Código Realizada

O projeto foi otimizado com as seguintes melhorias:

- ✅ **Dependências desnecessárias removidas**: `datasets`, `accelerate`, `tensorboard`, `opencv-python`, `pandas`, `wandb`, `gradio`, `spacy`
- ✅ **Código legado removido**: Funções não utilizadas (`_generate_answer`, `_generate_explanation`, `_analyze_image_features`)
- ✅ **Rotas duplicadas removidas**: Endpoints `/api/photo_game/*` duplicados foram limpos
- ✅ **Console.log removidos**: Logs de debug desnecessários removidos do frontend
- ✅ **Imports não utilizados**: Limpeza de imports desnecessários

### Atualizando o Dataset

Para adicionar novas imagens ao dataset:

1. Adicione as imagens na pasta `data/recife_historic/[nome_do_local]`
2. Execute o script de atualização:
   ```bash
   python update_photo_descriptions.py
   ```
3. Verifique se todos os caminhos estão corretos:
   ```bash
   python fix_photo_paths.py
   ```

### Solução de Problemas Comuns

- **Imagens não carregando**: Verifique se os caminhos no `photo_descriptions.json` começam com `data/`
- **Erros de classificação**: Verifique se as imagens estão nas pastas corretas
- **Problemas de desempenho**: Reduza o tamanho das imagens ou aumente os recursos do servidor
- **Erro ao importar NLTK**: Execute `python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"`

## 📊 Relatório Técnico

### Arquitetura do Modelo

O modelo implementa uma **CNN (Convolutional Neural Network)** personalizada para reconhecimento de arquitetura histórica:

```python
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=12):
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.09),
            nn.Linear(256, num_classes)
        )
```

### Processo de Treinamento

**Dataset:**
- 60 imagens reais (distribuídas em 12 classes)
- 12 classes (locais históricos)
- Data augmentation aplicada

Métrica recente (Transfer Learning ResNet18): ValAcc 100% (validação com 12 imagens).

**Hiperparâmetros:**
```python
{
    'epochs': 100,
    'batch_size': 2,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'dropout': 0.3,
    'optimizer': 'AdamW',
    'loss': 'CrossEntropyLoss'
}
```

**Métricas de Performance:**
- **Acurácia Inicial**: ~12%
- **Acurácia Final**: 96%
- **Épocas até Convergência**: ~89
- **Tempo de Treinamento**: ~3 minutos
- **Loss Final**: 0.97

### Técnicas de Deep Learning Utilizadas

#### 1. **Convoluções (Convolutions)**
```python
nn.Conv2d(in_channels, out_channels, kernel_size)
```
- **Função**: Detecta características visuais
- **Canais**: 3→64→128→256→512
- **Kernels**: 7×7, 3×3
- **Padding**: Preserva dimensões

#### 2. **Pooling (MaxPool2d)**
```python
nn.MaxPool2d(kernel_size, stride)
```
- **Função**: Reduz dimensionalidade
- **Efeito**: Extrai características mais importantes
- **Downsampling**: 4× no total

#### 3. **Dropout (Regularização)**
```python
nn.Dropout(0.3)
```
- **Função**: Previne overfitting
- **Taxa**: 30%, 15%, 9%
- **Efeito**: Melhora generalização

#### 4. **Data Augmentation**
```python
# Transformações aplicadas:
- RandomHorizontalFlip(0.2)  # Flip 20%
- ColorJitter(0.1)           # Cores variadas
- Resize(224×224)            # Tamanho fixo
```
- **Função**: Aumenta dataset artificialmente
- **Benefício**: Melhora robustez do modelo

#### 5. **Learning Rate Scheduling**
```python
ReduceLROnPlateau(factor=0.5, patience=5)
```
- **Função**: Ajusta taxa de aprendizado
- **Estratégia**: Reduz quando loss estaciona
- **Efeito**: Melhor convergência

#### 6. **Early Stopping**
```python
if accuracy > 95:
    print("Convergência alcançada!")
    break
```
- **Função**: Para treinamento antecipadamente
- **Meta**: 95%+ acurácia
- **Benefício**: Economiza tempo

### Pipeline de Predição

1. **Input**: Imagem (224×224×3)
2. **Preprocessing**: Normalização (mean=[0.485, 0.456, 0.406])
3. **Feature Extraction**: CNNs extraem características
4. **Classification**: Dense layers classificam
5. **Output**: Classe + Confiança

### Conceitos de Deep Learning

#### **Por que CNNs para Imagens?**
- **Invariância espacial**: Detecta padrões em qualquer posição
- **Hierarquia**: Características simples → complexas
- **Compartilhamento de pesos**: Eficiente

#### **Por que Deep Learning?**
- **Aprende automaticamente**: Não precisa features manuais
- **Escala**: Melhora com mais dados
- **Generaliza**: Funciona com novas imagens

#### **Como o modelo aprende?**
1. **Forward Pass**: Imagem → Predição
2. **Loss Calculation**: Compara com ground truth
3. **Backpropagation**: Calcula gradientes
4. **Weight Update**: Ajusta parâmetros via Adam
5. **Repete**: Até convergência

## 📁 Estrutura do Projeto

```
appDeepLearning/
├── 📄 main.py                          # Aplicação Flask principal
├── 📄 start_game.py                    # Script de inicialização
├── 📄 RETREINAR_MODELO.py              # Script de retreinamento
├── 📄 requirements.txt                 # Dependências
├── 📄 README.md                        # Este arquivo
│
├── 📁 models/
│   ├── improved_recife_historic_model.pth  # Modelo treinado
│   ├── multimodal_model.py             # (Legado)
│   └── trained_image_model.pth        # (Legado)
│
├── 📁 training/
│   ├── improved_recife_trainer.py      # Treinador melhorado ⭐
│   ├── recife_historic_trainer.py     # Treinador original
│   ├── image_trainer.py                # (Legado)
│   └── trainer.py                      # (Legado)
│
├── 📁 data/
│   └── recife_historic/                # Dataset de fotos reais
│       ├── marco_zero/
│       ├── casa_da_cultura/
│       ├── forte_das_cinco_pontas/
│       └── ... (outros locais)
│
├── 📁 game/
│   └── gamification.py                 # Sistema de gamificação
│
├── 📁 templates/
│   └── index.html                       # Interface web
│
├── 📁 static/                           # Arquivos estáticos
├── 📁 config/                          # Configurações
└── 📁 logs/                            # Logs do sistema
```

## 🔧 API Endpoints

### `/api/process_image` (POST)
Processa imagem e retorna identificação do local

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "question": "Que local histórico é este?",
  "player_id": "uuid-1234"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "Marco Zero",
  "explanation": "Praça principal do Recife...",
  "score": 10,
  "confidence": "0.95",
  "location_info": {
    "nome": "Marco Zero",
    "descricao": "Praça principal...",
    "historia": "Local onde a cidade foi fundada..."
  }
}
```

### `/api/leaderboard` (GET)
Retorna ranking de jogadores

### `/api/player_stats/<id>` (GET)
Retorna estatísticas do jogador

## 🎯 Próximas Melhorias

- [x] Transfer Learning com modelos pré-treinados (ResNet18)
- [x] Comparação visual por embeddings + similaridade do cosseno
- [ ] Interface mobile responsiva aprimorada
- [ ] Mais pontos históricos
- [ ] Modo offline
- [ ] Exportar relatórios de progresso
- [ ] Quiz interativo sobre história

## 📚 Referências de Deep Learning

### Conceitos Fundamentais

- **Neural Networks**: Redes de neurônios artificiais
- **CNNs**: Especializadas para imagens
- **Backpropagation**: Como a rede aprende
- **Gradient Descent**: Otimização de parâmetros
- **Overfitting**: Quando modelo decora dados
- **Regularization**: Dropout, weight decay
- **Data Augmentation**: Ampliar dataset

### Artigos e Tutoriais

1. **CNNs para Visão Computacional**: https://cs231n.github.io/
2. **PyTorch Documentation**: https://pytorch.org/docs/
3. **Deep Learning Book**: Goodfellow, Bengio, Courville

## 🤝 Contribuindo

Para contribuir:
1. Adicione fotos em `data/recife_historic/`
2. Retreine o modelo: `python RETREINAR_MODELO.py`
3. Teste e reporte melhorias

## 📄 Licença

Projeto educacional desenvolvido para estudo de Deep Learning aplicado.

---

**🏛️ Explore a História do Recife com IA!**
