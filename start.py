#!/usr/bin/env python3
"""
Script simples para iniciar o servidor
Usa o caminho correto automaticamente
"""

import os
import sys
import subprocess

def main():
    """Inicia o servidor Flask"""
    print("🚀 Iniciando servidor...")
    print("📍 Acesse: http://localhost:5000")
    print("⚠️  Pressione Ctrl+C para parar\n")
    
    # Executar main.py do diretório src
    main_path = os.path.join(os.path.dirname(__file__), 'src', 'main.py')
    
    try:
        subprocess.check_call([sys.executable, main_path])
    except KeyboardInterrupt:
        print("\n\n✅ Servidor encerrado")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
