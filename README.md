# 🏛️ Pontos Históricos do Recife - IA Educacional

Sistema educacional gamificado que utiliza **Deep Learning** para reconhecimento de pontos históricos do Recife através de imagens.

## 🚀 Início Rápido

```bash
# Clone o repositório
git clone [URL]
cd pontosTuristicosRecife

# Execute tudo com 1 comando!
python run.py
```

Acesse: **http://localhost:5000**

📖 **Guia completo**: [docs/INICIO_RAPIDO.md](docs/INICIO_RAPIDO.md)

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura](#-estrutura)
- [Modelo de IA](#-modelo-de-ia)
- [Dados e Gerenciamento](#-dados-e-gerenciamento)
- [API Endpoints](#-api-endpoints)
- [Solução de Problemas](#-solução-de-problemas)

---

## 🎯 Sobre o Projeto

Sistema de reconhecimento de imagens especializado em identificar **12 pontos históricos do Recife** usando **Redes Neurais Convolucionais (CNNs)** treinadas com 60 fotografias reais.

### 🎮 Modos de Jogo

1. **Modo Identificação** - Envie uma foto e receba informações históricas
2. **Modo Mistério** - Adivinhe o local através de dicas e ganhe XP
3. **Modo Foto** - Compare sua foto com fotos oficiais
4. **Modo Descrição** - Descreva a foto e ganhe pontos por precisão

---

## ✨ Funcionalidades

- 📸 **Reconhecimento de Imagens** - CNN treinada com PyTorch
- 🎮 **Gamificação** - XP, níveis, conquistas, ranking em tempo real
- �  **Modo Mistério** - Sistema de dicas progressivas (-5 XP por dica)
- � **SEducacional** - Informações históricas detalhadas
- 👤 **Sistema de Login** - Cadastro e autenticação
- ⚡ **Tempo Real** - Respostas instantâneas
- 📊 **Estatísticas** - Acompanhe seu progresso e conquistas
- 🔄 **Auto-reload** - Ranking atualiza automaticamente a cada 5 minutos
- 🎯 **Anti-repetição** - Sistema evita repetir as últimas 10 fotos

---

## 🛠️ Tecnologias

**Backend:** PyTorch, Flask, Python 3.8+  
**IA/NLP:** CNN (ResNet18), NLTK, scikit-learn  
**Frontend:** HTML/CSS/JavaScript, Bootstrap

---

## 📥 Instalação

### Requisitos
- Python 3.8+
- 4GB RAM mínimo
- 500MB espaço em disco

### Instalação Automática (Recomendado)

```bash
# 1. Clone
git clone [URL]
cd pontosTuristicosRecife

# 2. (Opcional) Ambiente virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Execute (instala tudo automaticamente)
python run.py
```

### Instalação Manual

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
python scripts/training/RETREINAR_MODELO.py
python src/main.py
```

---

## 🎮 Como Usar

### Iniciar

```bash
python run.py        # Completo (recomendado)
python start.py      # Apenas servidor
python src/main.py   # Direto
```

### Acessar

1. Abra: **http://localhost:5000**
2. Crie conta ou faça login
3. Escolha um modo de jogo
4. Comece a jogar!

**Parar:** Pressione `Ctrl+C`

---

## 📁 Estrutura do Projeto

```
pontosTuristicosRecife/
│
├── run.py                    # Script principal ⭐
├── start.py                  # Servidor rápido
├── requirements.txt
│
├── src/
│   └── main.py              # Aplicação Flask
│
├── scripts/
│   ├── training/
│   │   └── RETREINAR_MODELO.py
│   └── utils/
│       └── reset_players.py
│
├── training/
│   └── improved_recife_trainer.py
│
├── models/
│   └── improved_recife_historic_model.pth
│
├── data/
│   ├── recife_historic/     # 60 imagens (12 locais)
│   ├── users.json           # Usuários
│   └── players.json         # Jogadores
│
├── game/
│   ├── gamification.py
│   └── photo_description_game.py
│
├── templates/               # Interface web
│
└── docs/                    # Documentação
```

---

## 🤖 Modelo de IA

### Arquitetura CNN (Transfer Learning - ResNet18)

```
ImprovedCNN:
├── Feature Extractor
│   ├── Conv2d(3→64→128→256→512)
│   └── AdaptiveAvgPool2d(4×4)
│
├── Classifier
│   ├── Linear(8192→1024) + Dropout(0.3)
│   ├── Linear(1024→512) + Dropout(0.15)
│   ├── Linear(512→256) + Dropout(0.09)
│   └── Linear(256→12)  # 12 locais
│
└── Saída: Predição + Confiança
```

### Métricas

- **Parâmetros**: 13.7 milhões
- **Dataset**: 60 imagens (12 classes)
- **Acurácia**: 96% (experimental) | 100% (validação)
- **Treinamento**: 2-3 minutos
- **Técnicas**: Data Augmentation, Early Stopping, Dropout

### 🏛️ Locais Reconhecidos (12)

1. Casa da Cultura
2. Forte das Cinco Pontas
3. Igreja Madre de Deus
4. Igreja Nossa Senhora do Carmo
5. Igreja Santo Antônio
6. Igreja São Pedro dos Clérigos
7. Marco Zero
8. Mercado São José
9. Palácio da Justiça
10. Rua Aurora
11. Rua do Bom Jesus
12. Teatro Santa Isabel

---

## 💾 Dados e Gerenciamento

### Armazenamento

- **`data/users.json`** - Login, senha hash, email
- **`data/players.json`** - Pontos, níveis, conquistas
- **`data/photo_descriptions.json`** - Metadados das imagens

### Comandos

```bash
# Resetar jogadores
python scripts/utils/reset_players.py

# Treinar modelo
python scripts/training/RETREINAR_MODELO.py

# Adicionar fotos
# 1. Coloque em data/recife_historic/[local]/
# 2. Execute: python scripts/training/RETREINAR_MODELO.py

# Backup
cp data/users.json data/users.json.backup
cp data/players.json data/players.json.backup
```

---

## 📊 API Endpoints

### POST `/api/process_image`
Processa imagem e retorna identificação

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "question": "Que local histórico é este?",
  "player_id": "uuid-1234"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "Marco Zero",
  "explanation": "Praça principal...",
  "score": 10,
  "confidence": "0.95"
}
```

### Outros Endpoints

- `GET /api/leaderboard` - Ranking de jogadores
- `GET /api/player_stats/<id>` - Estatísticas do jogador
- `POST /api/create_player` - Criar novo jogador
- `POST /api/photo_game/mystery_guess` - Enviar palpite no Modo Mistério
- `GET /api/photo_game/random_photo` - Obter foto aleatória
- `POST /api/admin/reset_players` - Resetar ranking (admin)
- `POST /api/admin/reload_players` - Recarregar dados do arquivo
- `GET/POST /login` - Login
- `GET/POST /register` - Cadastro

---

## 🔧 Solução de Problemas

### Erros Comuns

**ModuleNotFoundError**
```bash
python start.py  # Execute da raiz
```

**Porta ocupada**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

**Modelo não encontrado**
```bash
python scripts/training/RETREINAR_MODELO.py
```

📖 **Guia completo**: [docs/SOLUCAO_PROBLEMAS.md](docs/SOLUCAO_PROBLEMAS.md)

---

## 🤝 Contribuindo

1. Adicione fotos em `data/recife_historic/[local]/`
2. Retreine: `python scripts/training/RETREINAR_MODELO.py`
3. Teste e reporte melhorias

---

## 📚 Documentação

- [Guia Rápido](docs/INICIO_RAPIDO.md)
- [Deploy em VM](docs/DEPLOY_VM.md)
- [Solução de Problemas](docs/SOLUCAO_PROBLEMAS.md)

---

## 📄 Licença

Projeto educacional para estudo de Deep Learning aplicado.

---

**🏛️ Explore a História do Recife com IA!**

Desenvolvido com ❤️ para educação e preservação da história cultural do Recife.
