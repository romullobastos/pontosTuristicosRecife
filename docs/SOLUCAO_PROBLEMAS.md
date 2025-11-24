# 🔧 Solução de Problemas Comuns

## ❌ Erro: "ModuleNotFoundError: No module named 'game'"

### Problema
```
ModuleNotFoundError: No module named 'game'
```

### Causa
O Python não está encontrando os módulos porque você está executando de um diretório diferente.

### Solução

#### ✅ Opção 1: Use o script start.py (RECOMENDADO)
```bash
python start.py
```

#### ✅ Opção 2: Use o run.py
```bash
python run.py
```

#### ✅ Opção 3: Execute da raiz do projeto
```bash
# Certifique-se de estar na raiz do projeto
cd pontosTuristicosRecife
python src/main.py
```

#### ❌ NÃO FAÇA:
```bash
cd src
python main.py  # ❌ Isso não funciona!
```

---

## ❌ Erro: "No module named 'torch'"

### Problema
```
ModuleNotFoundError: No module named 'torch'
```

### Causa
Dependências não instaladas.

### Solução
```bash
pip install -r requirements.txt
```

Ou use o script completo:
```bash
python run.py
```

---

## ❌ Erro: "No module named 'nltk'"

### Problema
```
ModuleNotFoundError: No module named 'nltk'
```

### Causa
NLTK não instalado ou recursos não baixados.

### Solução
```bash
# Instalar NLTK
pip install nltk

# Baixar recursos
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

Ou use o script completo:
```bash
python run.py
```

---

## ❌ Erro: "Address already in use"

### Problema
```
OSError: [Errno 98] Address already in use
```

### Causa
Porta 5000 já está em uso.

### Solução

#### Opção 1: Parar o processo existente
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

#### Opção 2: Mudar a porta
Edite `src/main.py` na última linha:
```python
# Antes
app.run(debug=True, host='0.0.0.0', port=5000)

# Depois
app.run(debug=True, host='0.0.0.0', port=5001)
```

---

## ❌ Erro: "Model file not found"

### Problema
```
FileNotFoundError: models/improved_recife_historic_model.pth
```

### Causa
Modelo não foi treinado ainda.

### Solução
```bash
# Treinar o modelo
python scripts/training/RETREINAR_MODELO.py
```

Ou use o script completo que treina automaticamente:
```bash
python run.py
```

---

## ❌ Erro: "Template not found"

### Problema
```
jinja2.exceptions.TemplateNotFound: login.html
jinja2.exceptions.TemplateNotFound: index.html
```

### Causa
Flask não está encontrando os templates porque o diretório de trabalho está errado.

### Solução ✅ (CORRIGIDO)
O problema foi corrigido no código! Agora o Flask encontra os templates automaticamente.

Execute normalmente:
```bash
python start.py
# ou
python run.py
# ou
python src/main.py
```

Se ainda tiver problemas, certifique-se de estar na raiz do projeto:
```bash
# Verificar diretório atual
pwd  # Linux/Mac
cd   # Windows

# Ir para raiz se necessário
cd pontosTuristicosRecife
```

---

## ❌ Erro: "Permission denied"

### Problema
```
PermissionError: [Errno 13] Permission denied
```

### Causa
Falta de permissões para criar/modificar arquivos.

### Solução

#### Windows
Execute o terminal como Administrador.

#### Linux/Mac
```bash
# Dar permissão de execução
chmod +x run.py
chmod +x start.py

# Ou executar com sudo (não recomendado)
sudo python run.py
```

---

## ❌ Imagens não carregam

### Problema
Imagens não aparecem na interface.

### Causa
Caminhos incorretos ou arquivos faltando.

### Solução
```bash
# Verificar se as imagens existem
ls data/recife_historic/

# Verificar caminhos no JSON
cat data/photo_descriptions.json
```

Certifique-se de que os caminhos começam com `data/`:
```json
{
  "image_path": "data/recife_historic/marco_zero/foto.jpg"
}
```

---

## ❌ Baixa acurácia do modelo

### Problema
Modelo não identifica corretamente os locais.

### Causa
Modelo precisa ser retreinado com mais dados.

### Solução
```bash
# 1. Adicione mais fotos em data/recife_historic/[local]/
# 2. Retreine o modelo
python scripts/training/RETREINAR_MODELO.py
```

---

## ❌ Erro de memória

### Problema
```
RuntimeError: CUDA out of memory
```

### Causa
GPU sem memória suficiente.

### Solução
Edite `scripts/training/RETREINAR_MODELO.py`:
```python
# Reduzir batch size
batch_size = 1  # Era 2 ou maior

# Ou forçar uso de CPU
device = 'cpu'
```

---

## 🆘 Ainda com problemas?

### Checklist Rápido

1. ✅ Está na raiz do projeto?
   ```bash
   pwd  # Linux/Mac
   cd   # Windows
   ```

2. ✅ Dependências instaladas?
   ```bash
   pip list | grep torch
   pip list | grep flask
   ```

3. ✅ Python 3.8+?
   ```bash
   python --version
   ```

4. ✅ Ambiente virtual ativado?
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

5. ✅ Arquivos no lugar certo?
   ```bash
   ls src/main.py
   ls run.py
   ls start.py
   ```

### Reset Completo

Se nada funcionar, faça um reset:

```bash
# 1. Limpar cache Python
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 2. Reinstalar dependências
pip uninstall -y -r requirements.txt
pip install -r requirements.txt

# 3. Retreinar modelo
python scripts/training/RETREINAR_MODELO.py

# 4. Executar
python run.py
```

---

## 📞 Comandos Úteis de Debug

```bash
# Ver versão do Python
python --version

# Ver pacotes instalados
pip list

# Ver processos na porta 5000
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # Linux/Mac

# Ver estrutura de diretórios
tree /F  # Windows
tree     # Linux/Mac

# Testar imports
python -c "import torch; print(torch.__version__)"
python -c "import flask; print(flask.__version__)"
python -c "from game.gamification import GamificationSystem; print('OK')"
```

---

**🏛️ Problemas resolvidos? Execute: `python run.py`**
