#!/usr/bin/env python3
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
