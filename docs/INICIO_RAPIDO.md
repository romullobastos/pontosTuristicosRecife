# 🚀 Guia Rápido

## Executar o Projeto

### Opção 1: Completo (Recomendado)
```bash
python run.py
```
Instala dependências, configura NLTK, treina modelo e inicia servidor.

### Opção 2: Apenas Servidor
```bash
python start.py
```
Inicia apenas o servidor (mais rápido).

### Opção 3: Direto
```bash
python src/main.py
```

## Acesse
**http://localhost:5000**

Pressione `Ctrl+C` para parar.

---

## Primeira Vez

```bash
# 1. Clone
git clone [URL]
cd pontosTuristicosRecife

# 2. (Opcional) Ambiente virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Execute
python run.py
```

---

## Comandos Úteis

```bash
# Treinar modelo
python scripts/training/RETREINAR_MODELO.py

# Adicionar fotos
# 1. Coloque em data/recife_historic/[local]/
# 2. Execute: python scripts/training/RETREINAR_MODELO.py
```

---

## Problemas Comuns

### ModuleNotFoundError
```bash
# Use sempre da raiz do projeto
python start.py
```

### Porta ocupada
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Modelo não encontrado
```bash
python scripts/training/RETREINAR_MODELO.py
```

---

## Dados de Usuários

Os dados são salvos em:
- **Usuários**: `data/users.json` (login, senha, email)
- **Jogadores**: `data/players.json` (pontos, níveis, conquistas)

Para resetar:
```bash
python scripts/utils/reset_players.py
```

---

## Estrutura

```
pontosTuristicosRecife/
├── run.py              # Execute este!
├── start.py            # Apenas servidor
├── src/main.py         # Aplicação Flask
├── scripts/            # Scripts utilitários
├── data/               # Dados e imagens
├── game/               # Lógica do jogo
├── training/           # Módulos de treinamento
├── models/             # Modelos treinados
└── templates/          # Interface web
```

---

**🏛️ Explore a História do Recife com IA!**
