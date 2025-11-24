# 🚀 Guia de Deploy na VM

## 📋 Pré-requisitos

- Ubuntu/Debian (ou similar)
- Python 3.8+
- Node.js e npm (para PM2)
- Git

---

## 🔧 Passo 1: Preparar o Servidor

### 1.1 Atualizar sistema
```bash
sudo apt update
sudo apt upgrade -y
```

### 1.2 Instalar Python e dependências
```bash
sudo apt install -y python3 python3-pip python3-venv git
```

### 1.3 Instalar Node.js e PM2
```bash
# Instalar Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Instalar PM2 globalmente
sudo npm install -g pm2
```

### 1.4 Liberar porta 5000
```bash
sudo ufw allow 5000/tcp
sudo ufw reload
```

---

## 📥 Passo 2: Clonar o Projeto

```bash
# Ir para diretório home
cd ~

# Clonar repositório
git clone https://github.com/romullobastos/pontosTuristicosRecife.git

# Entrar na pasta
cd pontosTuristicosRecife
```

---

## 🐍 Passo 3: Configurar Ambiente Python

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt

# Instalar Gunicorn
pip install gunicorn
```

---

## 📚 Passo 4: Configurar NLTK

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

## 🤖 Passo 5: Verificar/Treinar Modelo

```bash
# Se o modelo não existir, treinar
if [ ! -f "models/improved_recife_historic_model.pth" ]; then
    python scripts/training/RETREINAR_MODELO.py
fi
```

---

## 🚀 Passo 6: Iniciar com PM2

```bash
# Dar permissão de execução ao script
chmod +x deploy.sh

# Executar deploy
./deploy.sh
```

---

## 📊 Passo 7: Gerenciar com PM2

### Comandos úteis:

```bash
# Ver status
pm2 status

# Ver logs
pm2 logs recife-ia

# Ver logs em tempo real
pm2 logs recife-ia --lines 100

# Parar aplicação
pm2 stop recife-ia

# Reiniciar aplicação
pm2 restart recife-ia

# Remover aplicação
pm2 delete recife-ia

# Salvar configuração (auto-start)
pm2 save

# Configurar auto-start no boot
pm2 startup
# Copie e execute o comando que aparecer
```

---

## 🌐 Passo 8: Acessar Aplicação

### Local (na VM):
```
http://localhost:5000
```

### Remoto (de outro computador):
```
http://IP-DA-VM:5000
```

**Exemplo:**
```
http://192.168.1.100:5000
```

---

## 🔄 Passo 9: Atualizar Aplicação

```bash
# Entrar na pasta
cd ~/pontosTuristicosRecife

# Ativar ambiente virtual
source venv/bin/activate

# Puxar atualizações
git pull origin main

# Instalar novas dependências (se houver)
pip install -r requirements.txt

# Reiniciar com PM2
pm2 restart recife-ia
```

---

## 🔒 Passo 10: Configurar Nginx (Opcional)

### 10.1 Instalar Nginx
```bash
sudo apt install -y nginx
```

### 10.2 Criar configuração
```bash
sudo nano /etc/nginx/sites-available/recife-ia
```

### 10.3 Adicionar configuração:
```nginx
server {
    listen 80;
    server_name seu-dominio.com;  # ou IP da VM

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout para uploads
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
    
    # Limite de upload
    client_max_body_size 50M;
}
```

### 10.4 Ativar configuração
```bash
sudo ln -s /etc/nginx/sites-available/recife-ia /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 10.5 Liberar porta 80
```bash
sudo ufw allow 80/tcp
sudo ufw reload
```

Agora acesse: `http://IP-DA-VM` (sem :5000)

---

## 🐛 Solução de Problemas

### Erro: "Port 5000 already in use"
```bash
# Ver o que está usando a porta
sudo lsof -i :5000

# Matar processo
sudo kill -9 PID
```

### Erro: "Permission denied"
```bash
# Dar permissão aos scripts
chmod +x deploy.sh
chmod +x scripts/training/RETREINAR_MODELO.py
```

### Erro: "Module not found"
```bash
# Verificar se está no ambiente virtual
which python
# Deve mostrar: ~/pontosTuristicosRecife/venv/bin/python

# Se não estiver, ativar:
source venv/bin/activate
```

### Ver logs de erro:
```bash
# Logs do PM2
pm2 logs recife-ia --err

# Logs do sistema
tail -f logs/app.log
```

---

## 📝 Checklist de Deploy

- [ ] Sistema atualizado
- [ ] Python 3.8+ instalado
- [ ] Node.js e PM2 instalados
- [ ] Porta 5000 liberada
- [ ] Projeto clonado
- [ ] Ambiente virtual criado
- [ ] Dependências instaladas
- [ ] NLTK configurado
- [ ] Modelo treinado/verificado
- [ ] PM2 iniciado
- [ ] Aplicação acessível
- [ ] PM2 configurado para auto-start

---

## 🎯 Comandos Rápidos

```bash
# Deploy completo
cd ~/pontosTuristicosRecife && ./deploy.sh

# Ver status
pm2 status

# Ver logs
pm2 logs recife-ia

# Reiniciar
pm2 restart recife-ia

# Atualizar
cd ~/pontosTuristicosRecife && git pull && pm2 restart recife-ia
```

---

**🏛️ Aplicação rodando com sucesso!**
