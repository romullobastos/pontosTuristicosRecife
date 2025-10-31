# 📚 Resumo do Projeto - Chatbot Educacional

## 🎯 O Que É?

Sistema gamificado que ensina pontos históricos do Recife usando **Deep Learning** e **NLP**. 
Usuário interage de 3 formas:
- **Foto**: Compara fotos do mesmo local (Deep Learning)
- **Descrição**: Descreve fotos históricas (NLP)
- **Mistério**: Adivinha o local histórico

---

## 🔍 Problemática

**Problema**: Estudantes têm dificuldade em memorizar história de forma engajante.

**Solução**: Chatbot interativo com IA + gamificação (pontos, níveis, XP).

---

## 📊 Datasets

- **Imagens (estado atual no filesystem)**: 60 fotos distribuídas em 12 pastas de classes em `data/recife_historic/`.
- **Classes definidas**: 12 locais históricos (uma pasta por classe)
- **Descrições**: `data/photo_descriptions.json` com 23 registros (descrições oficiais e keywords)
- **Locais**: Casa da Cultura, Forte das Cinco Pontas, Igrejas, Marco Zero, etc.

**Estrutura JSON (photo_descriptions.json)**:
```json
{
  "id": "marco_zero",
  "name": "Marco Zero",
  "official_description": "...",
  "keywords": ["praça", "histórico", "colônia"],
  "difficulty": "Médio",
  "points": 15
}
```

---

## 🔄 Pipeline (Como Funciona)

### 1. Modo Foto
```
Usuário envia foto → Modelo CNN classifica → Compara com foto do desafio
   ↓
Mesma classe? SIM → Alta similaridade (≥80%)
              NÃO → Baixa similaridade (≤30%)
```

### 2. Modo Descrição  
```
Usuário escreve → NLP processa texto → Compara com descrição oficial
   ↓
TF-IDF + Cosseno → Score final (60% similaridade + 40% keywords)
```

### 3. Gamificação
```
Ação → Calcula pontos → Adiciona XP → Verifica level up → Salva em JSON
```

---

## 🤖 Modelos

### CNN (Deep Learning)
```
Arquitetura: ImprovedCNN
- 4 camadas de convolução (64, 128, 256, 512)
- 3 camadas fully connected (1024, 512, 256)
- Output: 12 classes (locais históricos)
- Parâmetros: 13.7 milhões
```

**Como compara fotos**:
1. Classifica foto do desafio: "Igreja" (confiança 0.85)
2. Classifica foto do usuário: "Forte" (confiança 0.90)  
3. **São diferentes?** → Similaridade = 10% (correto!)

### NLP (Processamento de Texto)
```
Vectorizer: TF-IDF (1000 features)
Similarity: Cosine Similarity
Score: (similaridade * 0.6) + (keywords * 0.4)
```

**Exemplo**:
- Usuário: "Igreja barroca com torre alta"
- Oficial: "Igreja barroca com torre alta, século XVIII"
- Similaridade: 85% → Score: 70%

---

## 📈 Avaliação

### CNN
- ✅ Mesmo local → Similaridade ≥80%
- ❌ Local diferente → Similaridade ≤30%
- Verificar: `pred_class1 == pred_class2`

### NLP
- Score ≥80%: Excelente (150% pontos)
- Score ≥60%: Muito bom (110% pontos)
- Score ≥40%: Bom (80% pontos)
- Score <40%: Tentativa (10% pontos)

---

## 🏗️ Arquitetura Simples

```
Frontend (HTML/JS) 
    ↓ HTTP
Backend (Flask)
    ↓
├─ CNN → Classificação de imagens
├─ NLP → Processamento de texto
└─ Gamificação → Pontos/XP/Níveis
    ↓
JSON (players.json)
```

**Fluxo de dados**:
```
Imagem → CNN → Classe + Confiança
Texto → NLP → Score de Similaridade
Resultado → Gamificação → Pontos + Estatísticas
```

---

## 💻 Tecnologias

- **Python** + Flask
- **PyTorch** (Deep Learning)
- **NLTK + sklearn** (NLP)
- **JavaScript** (Frontend)

---

## 🎮 Como Usar (POC)

1. Acesse: `http://localhost:5000/chatbot`
2. Login: Digite seu nome
3. Escolha modo:
   - **Foto**: Tire foto do local
   - **Descrição**: Descreva a foto
   - **Mistério**: Adivinhe o local
4. Ganhe pontos e suba de nível!

---

## 📝 Resultados Esperados

✅ Igreja vs Forte → ~10% similaridade  
✅ Igreja vs Igreja → ~85% similaridade  
✅ Marco Zero vs Marco Zero → ~90% similaridade  
✅ Descrição boa → Score ≥70%  
✅ Descrição ruim → Score <40%  

---

## 🔑 Conceitos Importantes

1. **TF-IDF**: Converte texto em vetor numérico
2. **Cosine Similarity**: Compara vetores (0-1)
3. **CNN**: Extrai features visuais de imagens
4. **Softmax**: Converte saída em probabilidades
5. **Gamificação**: Engaja usuário com sistema de pontos

---

## 📁 Arquivos Principais

```
main.py              → Backend Flask (rotas API)
chatbot.html         → Frontend (interface)
improved_recife_trainer.py  → Treinador CNN
photo_description_game.py    → Jogo NLP
gamification.py      → Sistema de pontos
```

---

## 🎓 Para Estudar

1. **Deep Learning**: Como CNN classifica imagens
2. **NLP**: Como TF-IDF compara textos
3. **Gamificação**: Como calcular XP/pontos
4. **Flask**: Como criar APIs REST
5. **JavaScript**: Como enviar/receber dados

**Foco**: Entender como CNN compara fotos e NLP avalia descrições!

