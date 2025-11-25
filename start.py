#!/usr/bin/env python3
"""
Script para iniciar o servidor com auto-reload (watchdog)
Recarrega automaticamente quando detecta mudanças nos arquivos
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    """Inicia o servidor Flask com watchdog"""
    print("🚀 Iniciando servidor com auto-reload...")
    print("📍 Acesse: http://localhost:5000")
    print("👀 Monitorando mudanças nos arquivos...")
    print("⚠️  Pressione Ctrl+C para parar\n")
    
    # Verificar se watchdog está instalado
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        watchdog_available = True
    except ImportError:
        print("⚠️  Watchdog não instalado. Rodando sem auto-reload.")
        print("   Para instalar: pip install watchdog\n")
        watchdog_available = False
    
    # Executar main.py do diretório src
    main_path = os.path.join(os.path.dirname(__file__), 'src', 'main.py')
    
    if not watchdog_available:
        # Rodar sem watchdog
        try:
            subprocess.check_call([sys.executable, main_path])
        except KeyboardInterrupt:
            print("\n\n✅ Servidor encerrado")
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            sys.exit(1)
    else:
        # Rodar com watchdog
        run_with_watchdog(main_path)

def run_with_watchdog(main_path):
    """Executa o servidor com monitoramento de arquivos"""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    
    class ChangeHandler(FileSystemEventHandler):
        def __init__(self):
            self.process = None
            self.last_restart = 0
            self.restart_delay = 1  # segundos
            
        def start_server(self):
            """Inicia o processo do servidor"""
            if self.process:
                self.stop_server()
            
            print("🔄 Iniciando servidor...")
            self.process = subprocess.Popen(
                [sys.executable, main_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Mostrar output do servidor em tempo real
            def print_output():
                for line in self.process.stdout:
                    print(line, end='')
            
            import threading
            threading.Thread(target=print_output, daemon=True).start()
        
        def stop_server(self):
            """Para o processo do servidor"""
            if self.process:
                print("\n🛑 Parando servidor...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.process = None
        
        def on_modified(self, event):
            """Callback quando arquivo é modificado"""
            if event.is_directory:
                return
            
            # Ignorar arquivos temporários e cache
            ignored_extensions = ['.pyc', '.pyo', '.pyd', '__pycache__', '.git', '.log']
            if any(ext in event.src_path for ext in ignored_extensions):
                return
            
            # Ignorar arquivos de dados
            if 'data/' in event.src_path or 'models/' in event.src_path:
                return
            
            # Debounce: evitar múltiplos restarts
            current_time = time.time()
            if current_time - self.last_restart < self.restart_delay:
                return
            
            self.last_restart = current_time
            
            print(f"\n📝 Detectada mudança em: {event.src_path}")
            print("🔄 Reiniciando servidor...\n")
            self.start_server()
    
    # Configurar observer
    handler = ChangeHandler()
    observer = Observer()
    
    # Monitorar diretórios
    watch_dirs = ['src', 'game', 'training', 'templates']
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    for watch_dir in watch_dirs:
        dir_path = os.path.join(root_dir, watch_dir)
        if os.path.exists(dir_path):
            observer.schedule(handler, dir_path, recursive=True)
            print(f"👀 Monitorando: {watch_dir}/")
    
    print()
    
    # Iniciar servidor
    handler.start_server()
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Encerrando...")
        handler.stop_server()
        observer.stop()
        observer.join()
        print("✅ Servidor encerrado")

if __name__ == "__main__":
    main()
