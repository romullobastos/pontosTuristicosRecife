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
    print("\nCarregando dataset...")
    trainer = ImprovedRecifeHistoricTrainer()
    
    print("\nVerificando modelo antigo...")
    import os
    if os.path.exists('models/improved_recife_historic_model.pth'):
        os.remove('models/improved_recife_historic_model.pth')
        print("✓ Modelo antigo removido")
    else:
        print("✓ Nenhum modelo antigo encontrado (primeiro treinamento)")
    
    print(f"\n✓ Locais encontrados: {trainer.num_classes}")
    print(f"✓ Total de imagens: {len(trainer.dataset)}")
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
        print("✓ MODELO SALVO COM SUCESSO!")
        print("=" * 70)
        print("\nO aplicativo vai usar automaticamente o novo modelo")
        print("Se o aplicativo estiver rodando, recarregue a pagina")
        print("=" * 70)
    else:
        print("\n✗ Falha no treinamento")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Treinamento interrompido pelo usuario")
    except Exception as e:
        print(f"\n\n✗ ERRO: {e}")
        import traceback
        traceback.print_exc()

