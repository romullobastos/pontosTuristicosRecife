# Projeto: Chatbot Gamificado para Ensino de Pontos Históricos do Recife

## 1. Definição do Projeto

Sistema educacional gamificado que utiliza Deep Learning para ensinar e avaliar conhecimentos sobre pontos históricos do Recife através de três modos de interação:
- **Modo Foto**: Comparação visual de fotos usando Deep Learning
- **Modo Descrição**: Avaliação de descrições usando NLP
- **Modo Mistério**: Desafio de identificação histórica

---

## 2. Problemática Escolhida

### Problema Principal
Como tornar o aprendizado sobre patrimônio histórico do Recife mais interativo, envolvente e eficaz, especialmente para estudantes e turistas que precisam aprender sobre os locais históricos da cidade?

### Questões Específicas
1. **Engajamento Educacional**: Estudantes têm dificuldade em memorizar informações históricas tradicionais
2. **Avaliação Objetiva**: Como avaliar o conhecimento adquirido de forma precisa e justa?
3. **Interatividade**: Como tornar o aprendizado mais dinâmico e menos monotônico?
4. **Tecnologia Educacional**: Integrar IA/Deep Learning com métodos de ensino tradicionais

### Solução Proposta
Sistema chatbot gamificado que combina:
- Análise visual de imagens com CNN
- Processamento de linguagem natural (NLP) para avaliação de descrições
- Gamificação (pontos, níveis, XP, conquistas)
- Interface interativa e intuitiva

---

## 3. Datasets Utilizados

### 3.1 Dataset de Imagens Históricas
- **Fonte**: `data/recife_historic/`
- **Estrutura**: 12 locais históricos diferentes
- **Total de Imagens**: 28 imagens
- **Locais**: Casa da Cultura, Forte das Cinco Pontas, Igreja Madre de Deus, Igreja Nossa Senhora do Carmo, Igreja Santo Antônio, Igreja São Pedro dos Clérigos, Marco Zero, Mercado São José, Palácio da Justiça, Rua Aurora, Rua do Bom Jesus, Teatro Santa Isabel

### 3.2 Dataset de Descrições
- **Fonte**: `data/photo_descriptions.json`
- **Formato**: JSON com descrições oficiais e keywords
- **Campos**:
  - `id`: Identificador único
  - `image_path`: Caminho da imagem
  - `name`: Nome do local
  - `official_description`: Descrição oficial detalhada
  - `keywords`: Lista de palavras-chave relevantes
  - `difficulty`: Nível de dificuldade (Fácil, Médio, Difícil)
  - `points`: Pontos associados

### 3.3 Recursos de NLP
- **Stopwords**: 207 palavras em português (NLTK)
- **Lemmatization**: WordNet (português)
- **Tokenization**: Regex-based (mais robusta que NLTK para português)

---

## 4. Pipeline/ETL

### 4.1 Fluxo de Processamento de Imagens

```
Usuário envia foto
    ↓
Decodificação Base64 → PIL Image
    ↓
Transforms (resize, normalize)
    ↓
Modelo CNN (ImprovedCNN)
    ↓
    ├─→ Predição de Classe (local histórico)
    └─→ Confiança (%)
```

### 4.2 Fluxo de Processamento de Texto (Modo Descrição)

```
Usuário escreve descrição
    ↓
Limpeza de texto (remove acentos, lowercase)
    ↓
Tokenization (regex: \b\w+\b)
    ↓
Remove stopwords
    ↓
Lemmatization (WordNet)
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity
    ↓
Score combinado (60% similaridade + 40% keywords)
```

### 4.3 Fluxo de Comparação Visual (Modo Foto)

```
Foto do Desafio + Foto do Usuário
    ↓
    ├─→ Classifica Foto do Desafio → Classe A
    └─→ Classifica Foto do Usuário → Classe B
    ↓
Compara Classes:
    ├─→ A == B: Alta similaridade
    └─→ A != B: Baixa similaridade
```

### 4.4 Pipeline de Gamificação

```
Ação do Usuário
    ↓
Calcula Pontos (baseado em score)
    ↓
Adiciona XP
    ↓
Verifica Level Up
    ↓
Atualiza Estatísticas
    ↓
Armazena em JSON (players.json)
```

---

## 5. Modelos Escolhidos

### 5.1 Modelo de Visão (CNN)

**Arquitetura**: `ImprovedCNN`

```python
Features (CNN):
  - Conv2d(3→64, kernel=7, stride=2)
  - MaxPool2d(kernel=3, stride=2)
  - Conv2d(64→128, kernel=3)
  - MaxPool2d(kernel=2)
  - Conv2d(128→256, kernel=3)
  - Conv2d(256→256, kernel=3)
  - MaxPool2d(kernel=2)
  - Conv2d(256→512, kernel=3)
  - Conv2d(512→512, kernel=3)
  - AdaptiveAvgPool2d(4x4)

Classifier:
  - Flatten
  - Linear(8192→1024) + ReLU + Dropout(0.5)
  - Linear(1024→512) + ReLU + Dropout(0.25)
  - Linear(512→256) + ReLU + Dropout(0.15)
  - Linear(256→num_classes)
```

**Total de Parâmetros**: 13,704,972  
**Classes**: 12 locais históricos  
**Input Size**: 224x224 RGB  
**Optimizer**: AdamW (lr=0.001, weight_decay=0.01)  
**Loss**: CrossEntropyLoss (label_smoothing=0.1)  
**Scheduler**: ReduceLROnPlateau

### 5.2 Modelo de NLP

**Arquitetura**: TF-IDF + Cosine Similarity

```python
Vectorizer: TfidfVectorizer(max_features=1000)
Similarity: cosine_similarity(matrix1, matrix2)
Score Final: (similarity * 0.6) + (keyword_score * 0.4)
```

**Preprocessamento**:
- Tokenization: `re.findall(r'\b\w+\b', text)`
- Stopword Removal: NLTK Portuguese (207 words)
- Lemmatization: WordNet Portuguese

---

## 6. Estratégias de Avaliação dos Modelos

### 6.1 Avaliação do Modelo de Visão

**Métricas Durante Treinamento**:
- Accuracy: Percentual de acertos na classificação
- Loss: CrossEntropyLoss com label smoothing
- Learning Rate: Ajustado dinamicamente (ReduceLROnPlateau)

**Critérios de Parada**:
- Early Stopping: Parar quando accuracy > 95%
- Convergence: Atingir alta accuracy
- Best Model: Salvar apenas melhor modelo

**Avaliação de Similaridade**:
```python
se (foto1.classe == foto2.classe):
    similarity = (confidence1 + confidence2) / 2
    se (confidence1 > 0.7 e confidence2 > 0.7):
        similarity *= 1.1  # Boost
senão:
    similarity = 0.1-0.3  # Baixa similaridade
```

### 6.2 Avaliação do Sistema de Descrições

**Métricas**:
- **Similarity Score**: TF-IDF cosine similarity (0-1)
- **Keyword Score**: Percentual de keywords encontradas (0-1)
- **Final Score**: (similarity * 0.6) + (keyword * 0.4)

**Thresholds**:
- Excelente: ≥ 80% → 150% pontos
- Muito Bom: ≥ 70% → 130% pontos
- Bom: ≥ 60% → 110% pontos
- Satisfatório: ≥ 40% → 80% pontos
- Tentativa: < 40% → 10% pontos (mínimo)

### 6.3 Avaliação de Gamificação

**Métricas de Engajamento**:
- Total de Tentativas
- Taxa de Acertos (%)
- Sequência de Acertos (Streak)
- Nível Alcançado
- Conquistas Desbloqueadas

**Sistema de Pontos**:
- Base: 10-50 pontos por desafio
- Multiplicadores: 0.5x a 1.5x
- Bônus de Sequência: Streak bonus
- Level Up: XP requirements

---

## 7. Arquitetura do Projeto

### 7.1 Arquitetura Geral

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         templates/chatbot.html                       │   │
│  │  - Interface interativa                              │   │
│  │  - Visualização de imagens                           │   │
│  │  - Chat em tempo real                                │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST API
┌────────────────────▼────────────────────────────────────────┐
│                     Backend (Flask)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              main.py (Rotas API)                     │   │
│  │  - /api/process_image                              │   │
│  │  - /api/compare_visual_similarity                  │   │
│  │  - /api/photo_game/submit_description              │   │
│  │  - /api/player_stats                               │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────┬───────────────────┬──────────────────────────┘
             │                   │
    ┌────────▼────────┐  ┌────────▼────────┐
    │   Deep Learning │  │   NLP System    │
    │                 │  │                 │
    │   models/       │  │   game/          │
    │   - CNN         │  │   - TF-IDF       │
    │   - Features    │  │   - Similarity    │
    └─────────────────┘  └─────────────────┘
             │                   │
    ┌────────▼───────────────────▼────────┐
    │       training/                     │
    │   - improved_recife_trainer.py       │
    │   - image_trainer.py                │
    └──────────────────────────────────────┘
             │
    ┌────────▼────────┐
    │   Gamification  │
    │                 │
    │   game/         │
    │   - Points      │
    │   - XP/Levels   │
    │   - Stats       │
    └─────────────────┘
```

### 7.2 Fluxo de Dados

```
┌──────────────┐
│   Usuário    │
└──────┬───────┘
       │
       ├─── Foto ──→ [CNN] ──→ Identificação ──→ Resposta
       │
       ├─── Foto ──→ [Compare] ──→ Similaridade ──→ Pontos
       │
       └─── Texto ──→ [NLP] ──→ Similaridade ──→ Pontos
```

### 7.3 Arquitetura do Modelo CNN

```
Input: [1, 3, 224, 224]
    ↓
┌─────────────────────┐
│   CNN Features      │
│  ┌───────────────┐  │
│  │ Conv + ReLU    │  │
│  │ Conv + ReLU    │  │
│  │ Pool           │  │
│  │ ...            │  │
│  └───────────────┘  │
│   Output: [1,8192] │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Classifier        │
│  ┌───────────────┐  │
│  │ Linear 8192   │  │
│  │ → 1024        │  │
│  │ Linear 1024   │  │
│  │ → 512         │  │
│  │ Linear 512    │  │
│  │ → 256         │  │
│  │ Linear 256    │  │
│  │ → 12 classes  │  │
│  └───────────────┘  │
└─────────────────────┘
    ↓
Output: [1, 12]  → Classe + Confiança
```

---

## 8. POC/Protótipo da Aplicação

### 8.1 Interface Principal

**URL**: http://localhost:5000/chatbot

**Características**:
- Login do jogador
- Dashboard com estatísticas (XP, Coins, Accuracy, Streak)
- Chat interativo
- Área de mensagens com scroll automático
- Upload de imagens

### 8.2 Funcionalidades Implementadas

#### ✅ Modo Foto (Comparação Visual)
- Upload de foto do usuário
- Comparação com foto do desafio usando Deep Learning
- Similaridade baseada em classificação de local histórico
- Sistema de pontos por similaridade

#### ✅ Modo Descrição (NLP)
- Visualização de foto histórica
- Descrição do usuário
- Avaliação usando TF-IDF + Cosseno
- Feedback com similaridade e keywords

#### ✅ Modo Mistério (Identificação)
- Desafio sem ver foto completa
- Adivinhação do local histórico
- Dicas disponíveis
- Bônus de 2x pontos para acerto

#### ✅ Gamificação
- Sistema de pontos e XP
- Níveis de progresso
- Sequências de acertos (Streak)
- Estatísticas de performance
- Conquistas e rankings

### 8.3 Tecnologias Utilizadas

**Backend**:
- Python 3.x
- Flask (framework web)
- PyTorch (Deep Learning)
- NLTK + sklearn (NLP)
- PIL (Processamento de imagens)

**Frontend**:
- HTML5 + CSS3
- JavaScript (Vanilla)
- Canvas API (visualização de imagens)
- LocalStorage (persistência de jogador)

**Modelos**:
- ImprovedCNN (PyTorch)
- TF-IDF Vectorizer (sklearn)
- Cosine Similarity (sklearn)

### 8.4 Dados de Teste

**Dataset**:
- 12 locais históricos do Recife
- 28 imagens no total
- Descrições oficiais completas
- Keywords para cada local

**Exemplo de Uso**:
1. Usuário escolhe "Modo Foto"
2. Sistema mostra foto do Marco Zero
3. Usuário tira foto do Marco Zero
4. Sistema compara e dá 95% de similaridade
5. Usuário ganha 15 pontos

---

## 9. Resultados Esperados

### 9.1 Métricas de Sucesso

**Técnicas**:
- Accuracy do modelo CNN ≥ 85%
- Similaridade visual: ≥80% para mesmos locais
- Similaridade visual: ≤30% para locais diferentes
- Similaridade textual: Correlação com qualidade da descrição

**Educacionais**:
- Engajamento: Aumento de tempo de estudo
- Retenção: Melhor memorização de informações históricas
- Motivação: Sistema de pontos e níveis

### 9.2 Diferenciais do Projeto

1. **Multimodal**: Combina visão (Deep Learning) e linguagem (NLP)
2. **Gamificação**: Sistema completo de pontos, níveis e conquistas
3. **Interatividade**: Três modos diferentes de aprendizado
4. **Tecnologia**: CNN customizada + NLP com TF-IDF
5. **Português**: Otimizado para língua portuguesa

---

## 10. Conclusão

Este projeto demonstra a aplicação prática de Deep Learning e NLP para educação, criando uma experiência gamificada que torna o aprendizado sobre patrimônio histórico mais envolvente e eficaz. A arquitetura multimodal permite múltiplas formas de interação, enquanto a gamificação mantém o engajamento do usuário.

---

## 📁 Estrutura de Arquivos

```
appDeepLearning/
├── data/
│   ├── recife_historic/        # Imagens de locais históricos
│   └── photo_descriptions.json # Descrições oficiais
├── training/
│   ├── improved_recife_trainer.py  # Treinador principal
│   ├── image_trainer.py
│   └── recife_historic_trainer.py
├── models/
│   └── improved_recife_historic_model.pth  # Modelo treinado
├── game/
│   ├── photo_description_game.py  # Jogo de descrições
│   └── gamification.py             # Sistema de gamificação
├── templates/
│   ├── chatbot.html               # Interface do chatbot
│   └── index.html                 # Interface original
├── main.py                        # Backend Flask
└── requirements.txt               # Dependências
```

---

**Desenvolvido com**: Python, PyTorch, Flask, NLTK, sklearn  
**Data**: 2025  
**Autor**: Sistema de Deep Learning Educacional

