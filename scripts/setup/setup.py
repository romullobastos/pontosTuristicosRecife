#!/usr/bin/env python3
"""
Script de inicialização do Jogo Educacional Multimodal
"""

import os
import sys
import subprocess
import json

def create_directories():
    """Cria os diretórios necessários"""
    directories = [
        'data',
        'data/recife_historic',
        'models',
        'training',
        'game',
        'utils',
        'config',
        'templates',
        'static',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Diretório criado: {directory}")

def create_sample_data():
    """Cria dados de exemplo para demonstração"""
    sample_images = [
        {
            "image_path": "data/recife_historic/marco_zero/marco-zero-6-1030x579.jpeg",
            "question": "Que local histórico é este?",
            "answer": "Este é o Marco Zero do Recife",
            "explanation": "O Marco Zero é a praça principal do Recife Antigo, onde a cidade foi fundada em 1537. É um marco histórico que representa o início da colonização portuguesa no Nordeste.",
            "category": "historic_location",
            "difficulty": "medium",
            "points": 20
        },
        {
            "image_path": "data/recife_historic/casa_da_cultura/bdf-20250516-070544-8be936-750x400.jpeg",
            "question": "Que local histórico é este?",
            "answer": "Esta é a Casa da Cultura",
            "explanation": "A Casa da Cultura é um importante centro cultural do Recife, localizado no antigo prédio da Casa de Detenção, construído em 1855. Foi transformada em centro cultural em 1976.",
            "category": "historic_location",
            "difficulty": "medium",
            "points": 20
        },
        {
            "image_path": "data/recife_historic/forte_das_cinco_pontas/museu-do-recife-forte-das-cinco-pontas-foto-arthur-de-souza-31.jpg",
            "question": "Que local histórico é este?",
            "answer": "Este é o Forte das Cinco Pontas",
            "explanation": "O Forte das Cinco Pontas é uma fortaleza histórica construída pelos holandeses em 1630. Tem formato pentagonal único e foi construída para defender a cidade. Hoje abriga o Museu da Cidade do Recife.",
            "category": "historic_location",
            "difficulty": "hard",
            "points": 25
        }
    ]
    
    # Salvar dados de exemplo
    with open('data/annotations.json', 'w', encoding='utf-8') as f:
        json.dump(sample_images, f, ensure_ascii=False, indent=2)
    
    print("✅ Dados de exemplo criados em data/annotations.json")

def create_config_file():
    """Cria arquivo de configuração"""
    config = {
        "model": {
            "vocab_size": 10000,
            "num_classes": 1000,
            "hidden_dim": 768,
            "max_length": 50
        },
        "training": {
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 0.001,
            "device": "auto"
        },
        "game": {
            "max_players": 100,
            "daily_challenges_reset": "00:00",
            "achievement_notifications": True
        },
        "api": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": True
        }
    }
    
    with open('config/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ Arquivo de configuração criado em config/config.json")

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    try:
        import torch
        import torchvision
        import flask
        import PIL
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        import tqdm
        print("✅ Todas as dependências principais estão instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        return False

def install_dependencies():
    """Instala as dependências necessárias"""
    print("📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False

def create_startup_script():
    """Cria script de inicialização"""
    startup_content = """#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    print("🎮 Iniciando Jogo Educacional Multimodal...")
    
    # Verificar se estamos no diretório correto
    if not os.path.exists('main.py'):
        print("❌ Execute este script no diretório raiz do projeto")
        sys.exit(1)
    
    # Verificar dependências
    try:
        import torch
        import flask
        print("✅ Dependências OK")
    except ImportError:
        print("❌ Instale as dependências primeiro: pip install -r requirements.txt")
        sys.exit(1)
    
    # Executar aplicação
    print("🚀 Iniciando servidor...")
    os.system("python main.py")

if __name__ == "__main__":
    main()
"""
    
    with open('start_game.py', 'w', encoding='utf-8') as f:
        f.write(startup_content)
    
    # Tornar executável no Unix
    if os.name != 'nt':
        os.chmod('start_game.py', 0o755)
    
    print("✅ Script de inicialização criado: start_game.py")

def main():
    """Função principal de inicialização"""
    print("🎮 Configurando Jogo Educacional Multimodal")
    print("=" * 50)
    
    # Criar estrutura de diretórios
    print("\n📁 Criando estrutura de diretórios...")
    create_directories()
    
    # Criar dados de exemplo
    print("\n📊 Criando dados de exemplo...")
    create_sample_data()
    
    # Criar arquivo de configuração
    print("\n⚙️ Criando arquivo de configuração...")
    create_config_file()
    
    # Verificar dependências
    print("\n🔍 Verificando dependências...")
    if not check_dependencies():
        print("\n📦 Instalando dependências...")
        if not install_dependencies():
            print("❌ Falha na instalação das dependências")
            return False
    
    # Criar script de inicialização
    print("\n🚀 Criando script de inicialização...")
    create_startup_script()
    
    print("\n" + "=" * 50)
    print("✅ Configuração concluída com sucesso!")
    print("\n📋 Próximos passos:")
    print("1. Execute: python start_game.py")
    print("2. Acesse: http://localhost:5000")
    print("3. Crie um jogador e comece a jogar!")
    print("\n🎯 Funcionalidades disponíveis:")
    print("• Reconhecimento de imagens")
    print("• Processamento de perguntas em português")
    print("• Explicações educativas")
    print("• Sistema de gamificação completo")
    print("• Ranking e conquistas")
    print("• Interface web moderna")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

