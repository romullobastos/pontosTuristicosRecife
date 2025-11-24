#!/bin/bash
# Script de Deploy para Produção com PM2

echo "🚀 Iniciando deploy..."

# 1. Verificar se está no ambiente virtual
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Ativando ambiente virtual..."
    source venv/bin/activate
fi

# 2. Atualizar código
echo "📥 Atualizando código..."
git pull origin main

# 3. Instalar dependências
echo "📦 Instalando dependências..."
pip install -r requirements.txt
pip install gunicorn

# 4. Configurar NLTK
echo "📚 Configurando NLTK..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 5. Verificar modelo
if [ ! -f "models/improved_recife_historic_model.pth" ]; then
    echo "⚠️  Modelo não encontrado! Treinando..."
    python scripts/training/RETREINAR_MODELO.py
fi

# 6. Criar diretórios necessários
mkdir -p logs
mkdir -p data

# 7. Verificar se PM2 está instalado
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 não está instalado!"
    echo "Instale com: sudo npm install -g pm2"
    exit 1
fi

# 8. Parar aplicação antiga (se existir)
echo "🛑 Parando aplicação antiga..."
pm2 delete recife-ia 2>/dev/null || true

# 9. Iniciar com PM2
echo "✅ Iniciando servidor com PM2..."
pm2 start venv/bin/gunicorn \
    --name recife-ia \
    --interpreter none \
    -- -w 4 -b 0.0.0.0:5000 --timeout 120 src.main:app

# 10. Salvar configuração PM2
pm2 save

# 11. Configurar auto-start (primeira vez)
if [ ! -f ~/.pm2-startup-configured ]; then
    echo "⚙️  Configurando auto-start do PM2..."
    pm2 startup
    touch ~/.pm2-startup-configured
    echo "⚠️  Execute o comando acima para configurar auto-start!"
fi

echo ""
echo "🎉 Deploy concluído!"
echo "📍 Servidor rodando em: http://0.0.0.0:5000"
echo ""
echo "📊 Comandos úteis:"
echo "  pm2 status          - Ver status"
echo "  pm2 logs recife-ia  - Ver logs"
echo "  pm2 restart recife-ia - Reiniciar"
echo "  pm2 stop recife-ia  - Parar"
