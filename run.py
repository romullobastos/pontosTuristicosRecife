#!/usr/bin/env python3
"""
Script único para executar todo o projeto
Instala dependências, configura NLTK, treina modelo e inicia servidor
"""

import os
import sys
import subprocess
import time

def print_step(step, message):
    """Imprime mensagem formatada"""
    print(f"\n{'='*60}")
    print(f"[PASSO {step}] {message}")
    print('='*60)

def check_venv():
    """Verifica se está em ambiente virtual"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if not in_venv:
        print("⚠️  AVISO: Você não está em um ambiente virtual!")
        print("   Recomendamos criar um: python -m venv venv")
        response = input("   Continuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            sys.exit(0)

def install_dependencies():
    """Instala dependências do requirements.txt"""
    print_step(1, "INSTALANDO DEPENDÊNCIAS")
    
    if not os.path.exists('requirements.txt'):
        print("❌ Arquivo requirements.txt não encontrado!")
        return False
    
    try:
        print("📦 Instalando pacotes Python...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
        ])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def setup_nltk():
    """Baixa recursos necessários do NLTK"""
    print_step(2, "CONFIGURANDO NLTK")
    
    try:
        import nltk
        print("📚 Baixando recursos do NLTK...")
        
        resources = ['stopwords', 'punkt', 'wordnet']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                print(f"   ✅ {resource}")
            except Exception as e:
                print(f"   ⚠️  {resource} - {e}")
        
        print("✅ NLTK configurado!")
        return True
    except ImportError:
        print("⚠️  NLTK não instalado, pulando...")
        return True
    except Exception as e:
        print(f"❌ Erro ao configurar NLTK: {e}")
        return False

def check_model():
    """Verifica se o modelo existe, senão treina"""
    print_step(3, "VERIFICANDO MODELO")
    
    model_path = 'models/improved_recife_historic_model.pth'
    
    if os.path.exists(model_path):
        print(f"✅ Modelo encontrado: {model_path}")
        response = input("   Deseja retreinar o modelo? (s/n): ")
        if response.lower() != 's':
            return True
    else:
        print("⚠️  Modelo não encontrado!")
        print("   Iniciando treinamento...")
    
    # Treinar modelo
    try:
        print("\n🎓 Treinando modelo (pode levar 2-3 minutos)...")
        if os.path.exists('scripts/training/RETREINAR_MODELO.py'):
            subprocess.check_call([sys.executable, "scripts/training/RETREINAR_MODELO.py"])
        elif os.path.exists('scripts/training/train_improved_model.py'):
            subprocess.check_call([sys.executable, "scripts/training/train_improved_model.py"])
        else:
            print("❌ Script de treinamento não encontrado!")
            return False
        
        print("✅ Modelo treinado com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao treinar modelo: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Treinamento interrompido pelo usuário")
        return False

def start_server():
    """Inicia o servidor Flask"""
    print_step(4, "INICIANDO SERVIDOR")
    
    if not os.path.exists('src/main.py'):
        print("❌ Arquivo src/main.py não encontrado!")
        return False
    
    try:
        print("🚀 Iniciando servidor Flask...")
        print("📍 Acesse: http://localhost:5000")
        print("⚠️  Pressione Ctrl+C para parar o servidor\n")
        
        time.sleep(1)
        
        # Iniciar servidor
        subprocess.check_call([sys.executable, "src/main.py"])
        
    except KeyboardInterrupt:
        print("\n\n✅ Servidor encerrado pelo usuário")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao iniciar servidor: {e}")
        return False

def main():
    """Função principal"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🏛️  APLICATIVO EDUCACIONAL - PONTOS HISTÓRICOS DO RECIFE ║
    ║                                                           ║
    ║  Script de Execução Completa                             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Verificar ambiente virtual
    check_venv()
    
    # Passo 1: Instalar dependências
    if not install_dependencies():
        print("\n❌ Falha na instalação de dependências")
        sys.exit(1)
    
    # Passo 2: Configurar NLTK
    if not setup_nltk():
        print("\n⚠️  Aviso: NLTK não configurado completamente")
    
    # Passo 3: Verificar/treinar modelo
    if not check_model():
        print("\n❌ Falha na verificação/treinamento do modelo")
        response = input("Deseja continuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            sys.exit(1)
    
    # Passo 4: Iniciar servidor
    start_server()
    
    print("\n✅ Execução concluída!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
