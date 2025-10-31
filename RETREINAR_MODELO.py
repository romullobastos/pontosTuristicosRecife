#!/usr/bin/env python3
"""
Script para RETREINAR o modelo com novas fotos

COMO USAR:
1. Adicione suas novas fotos em data/recife_historic/[nome_do_local]/
2. Execute este script: python RETREINAR_MODELO.py
3. Aguarde o treinamento terminar (vai mostrar o progresso)
"""
import sys
sys.path.append('training')

from improved_recife_trainer import ImprovedRecifeHistoricTrainer

def main():
    print("=" * 70)
    print("RETREINANDO MODELO - Pontos Historicos do Recife")
    print("=" * 70)
    
    # Criar treinador
    print("\nCarregando dataset (Transfer Learning: ResNet18)...")
    trainer = ImprovedRecifeHistoricTrainer(use_transfer=True)
    
    print("\nVerificando modelo antigo...")
    import os
    if os.path.exists('models/improved_recife_historic_model.pth'):
        os.remove('models/improved_recife_historic_model.pth')
        print("[OK] Modelo antigo removido")
    else:
        print("[OK] Nenhum modelo antigo encontrado (primeiro treinamento)")
    
    print(f"\n[OK] Locais encontrados: {trainer.num_classes}")
    print(f"[OK] Total de imagens: {len(trainer.dataset)}")
    print("\n" + "=" * 70)
    print("INICIANDO TREINAMENTO")
    print("=" * 70)
    print("\nConfiguracao:")
    print("  - Epocas: até convergência (ou máximo 200)")
    print("  - Batch size: 2")
    print("\nO treinamento vai parar quando chegar a 96% de acuracia")
    print("Isso geralmente leva cerca de 2-3 minutos")
    print("\n" + "-" * 70 + "\n")
    
    # Treinar
    model = trainer.train(epochs=200, batch_size=2)
    
    if model:
        print()
        print("=" * 70)
        print("SALVANDO MODELO...")
        trainer.save_model()
        print("=" * 70)
        print("[OK] MODELO SALVO COM SUCESSO!")
        print("=" * 70)
        print("\nO aplicativo vai usar automaticamente o novo modelo")
        print("Se o aplicativo estiver rodando, recarregue a pagina")
        print("=" * 70)
    else:
        print("\n[ERRO] Falha no treinamento")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[ATENCAO] Treinamento interrompido pelo usuario")
    except Exception as e:
        print(f"\n\n[ERRO] {e}")
        import traceback
        traceback.print_exc()

