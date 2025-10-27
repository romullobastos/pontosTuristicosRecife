# 📤 Como Fazer Commit para o GitHub

## ⚠️ IMPORTANTE: Git não está instalado

Você precisa instalar o Git primeiro. Siga as instruções abaixo:

### 1️⃣ Instalar Git

**Windows:**
1. Acesse: https://git-scm.com/download/win
2. Baixe e instale o Git
3. Reinicie o terminal/PowerShell

**Ou via winget:**
```powershell
winget install Git.Git
```

### 2️⃣ Depois de Instalar, Execute:

```powershell
# 1. Configurar Git (primeira vez)
git config --global user.name "Seu Nome"
git config --global user.email "seuemail@example.com"

# 2. Inicializar repositório
cd c:\Users\romul\Documents\projetos\appDeepLearning
git init

# 3. Adicionar remote
git remote add origin https://github.com/romullobastos/pontosTuristicosRecife.git

# 4. Adicionar arquivos
git add .

# 5. Fazer commit
git commit -m "feat: Sistema de reconhecimento de pontos históricos do Recife com Deep Learning

- Implementa CNN personalizada para reconhecimento de 12 pontos históricos
- Acurácia de 96% com dataset de 25 imagens reais
- Sistema de gamificação integrado
- Interface web com Flask
- Documentação completa incluída"

# 6. Fazer push
git branch -M main
git push -u origin main
```

### 3️⃣ Estrutura do Commit

**Arquivos incluídos:**
- ✅ Código-fonte completo
- ✅ Modelo treinado (`models/improved_recife_historic_model.pth`)
- ✅ Documentação (README.md + RELATORIO_DEEP_LEARNING.md)
- ✅ Scripts de treinamento
- ✅ Interface web (templates/index.html)
- ✅ Configurações e dependências

**Arquivos EXCLUÍDOS** (.gitignore):
- ❌ Imagens do dataset (muito grandes)
- ❌ __pycache__/
- ❌ Arquivos temporários
- ❌ Logs

### 4️⃣ Comandos Resumidos

```bash
git init
git remote add origin https://github.com/romullobastos/pontosTuristicosRecife.git
git add .
git commit -m "feat: Sistema de reconhecimento de pontos históricos do Recife"
git push -u origin main
```

### 5️⃣ Estrutura Esperada no GitHub

```
pontosTuristicosRecife/
├── README.md                    # Documentação principal
├── RELATORIO_DEEP_LEARNING.md   # Relatório técnico
├── RETREINAR_MODELO.py         # Script de retreinamento
├── main.py                      # Aplicação Flask
├── start_game.py                # Launcher
├── .gitignore                   # Arquivos ignorados
├── requirements.txt             # Dependências
├── models/
│   └── improved_recife_historic_model.pth
├── training/
│   ├── improved_recife_trainer.py
│   └── recife_historic_trainer.py
├── game/
│   └── gamification.py
├── templates/
│   └── index.html
└── data/
    └── recife_historic/         # (estrutura vazia no repo)
```

---

## 🚀 Alternativa Rápida: GitHub Desktop

Se preferir interface gráfica:

1. Instale: https://desktop.github.com/
2. Abra GitHub Desktop
3. File → Add Local Repository
4. Selecione: `c:\Users\romul\Documents\projetos\appDeepLearning`
5. Clique em "Publish repository"
6. Envie para: `romullobastos/pontosTuristicosRecife`

---

## 📝 Próximos Passos

Depois de fazer o commit:

1. ✅ Repositório estará no GitHub
2. ✅ Pode compartilhar o link
3. ✅ Outros podem clonar e usar
4. ✅ Faça updates quando adicionar mais fotos

**Link após push:** https://github.com/romullobastos/pontosTuristicosRecife

