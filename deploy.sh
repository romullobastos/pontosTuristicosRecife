#!/bin/bash
# Script de Deploy para Produção

echo "🚀 Iniciando deploy..."

# 1. Atualizar código
echo "📥 Atualizando código..."
git pull origin main

# 2. Instalar dependências
echo "📦 Instalando dependências..."
pip install -r requirements.txt

# 3. Configurar NLTK
echo "📚 Configurando NLTK..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 4. Verificar modelo
if [ ! -f "models/improved_recife_historic_model.pth" ]; then
    echo "⚠️  Modelo não encontrado! Treinando..."
    python scripts/training/RETREINAR_MODELO.py
fi

# 5. Criar diretórios necessários
mkdir -p logs
mkdir -p data

# 6. Parar servidor antigo (se existir)
echo "🛑 Parando servidor antigo..."
pkill -f "gunicorn.*src.main:app" || true

# 7. Iniciar servidor
echo "✅ Iniciando servidor..."
nohup gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 src.main:app > logs/gunicorn.log 2>&1 &

echo "🎉 Deploy concluído!"
echo "📍 Servidor rodando em: http://0.0.0.0:5000"
echo "📋 Logs em: logs/gunicorn.log"
