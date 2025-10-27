# 📦 Instalação de Bibliotecas NLP

Para usar recursos avançados de Processamento de Linguagem Natural no sistema de descrições de fotos, você precisa instalar as bibliotecas necessárias.

## 🚀 Instalação Rápida

### 1. Instalar bibliotecas Python

```bash
pip install -r requirements.txt
```

### 2. Download dos recursos do NLTK

O sistema tentará baixar automaticamente os recursos, mas se houver problemas, execute:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Ou execute o script de instalação:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3. (Opcional) Instalar spaCy

Se quiser usar modelos mais avançados:

```bash
pip install spacy
python -m spacy download pt_core_news_sm
```

## 📊 O que mudou?

### **Antes (Básico)**
- ❌ SequenceMatcher simples
- ❌ Comparação literal de strings
- ❌ Muito sensível a pequenas diferenças

### **Agora (NLP)**
- ✅ **Tokenização**: Divide texto em palavras
- ✅ **Remoção de Stopwords**: Remove palavras sem sentido (de, a, o, etc.)
- ✅ **Lemmatização**: Reduz palavras à raiz (ex: "históricos" → "histórico")
- ✅ **TF-IDF**: Pesa palavras importantes
- ✅ **Similaridade de Cosseno**: Compara semanticamente, não literalmente

## 🎯 Benefícios

1. **Entende sinônimos**: "antigo" e "histórico" são tratados de forma similar
2. **Ignora palavras vazias**: Foca no conteúdo importante
3. **Mais justo**: Diferentes formas de expressar a mesma ideia têm pontuação alta
4. **Melhor feedback**: Usuários não precisam copiar exatamente a descrição oficial

## 🧪 Teste

Depois de instalar, execute:

```bash
python main.py
```

Agora o sistema vai usar NLP automaticamente!

## ⚠️ Troubleshooting

Se houver erro "Resource not found":
```bash
python -m nltk.downloader all
```

Se houver problemas com stopwords em português:
```bash
python -m nltk.downloader stopwords
```
