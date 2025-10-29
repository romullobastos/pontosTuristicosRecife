import os
import sys
sys.path.append('training')

from improved_recife_trainer import ImprovedRecifeHistoricTrainer, AdvancedImageTransforms

def main():
    print("Iniciando treinamento do modelo MELHORADO de pontos históricos do Recife...")
    print("=" * 70)
    
    # Verificar se existe dataset
    if not os.path.exists('data/recife_historic'):
        print("Dataset não encontrado!")
        print("Execute primeiro: python setup_recife_images.py")
        return
    
    # Inicializar treinador melhorado
    # Use Transfer Learning para melhores resultados em dataset pequeno
    trainer = ImprovedRecifeHistoricTrainer(use_transfer=True)
    
    # Verificar se já existe modelo melhorado
    if trainer.load_model():
        print("Modelo melhorado já existe!")
        print("Para retreinar, delete o arquivo: models/improved_recife_historic_model.pth")
        
        # Testar modelo com algumas imagens
        print("\nTestando modelo melhorado...")
        test_images = [
            'data/recife_historic/casa_da_cultura/27b3dee28e68f2f29c3aa48729db4f06.jpg',
            'data/recife_historic/palacio_da_justica/entrada.jpg',
            'data/recife_historic/marco_zero/marco_zero_0.jpg',
            'data/recife_historic/praia_boa_viagem/praia_boa_viagem_0.jpg'
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                predicted, confidence = trainer.predict(img_path)
                print(f"{img_path}: {predicted} (confiança: {confidence:.2f})")
        
        return
    
    # Treinar modelo melhorado
    print("Iniciando treinamento melhorado...")
    print("Técnicas aplicadas:")
    print("- Data Augmentation avançado")
    print("- Transfer Learning (ResNet18 pré-treinada) + Fine-tuning da cabeça")
    print("- Balanceamento por WeightedRandomSampler")
    print("- Dropout/Regularização (na cabeça)")
    print("- Learning Rate Scheduler (ReduceLROnPlateau)")
    print("- Gradient Clipping")
    print("- Label Smoothing leve")
    print("- Early Stopping")
    
    model = trainer.train(epochs=80, batch_size=8)
    
    if model:
        # Salvar modelo melhorado
        trainer.save_model()
        print("\nTreinamento melhorado concluído e modelo salvo!")
        
        # Testar modelo treinado
        print("\nTestando modelo treinado...")
        test_images = [
            'data/recife_historic/casa_da_cultura/27b3dee28e68f2f29c3aa48729db4f06.jpg',
            'data/recife_historic/palacio_da_justica/entrada.jpg',
            'data/recife_historic/marco_zero/marco_zero_0.jpg',
            'data/recife_historic/praia_boa_viagem/praia_boa_viagem_0.jpg'
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                predicted, confidence = trainer.predict(img_path)
                print(f"{img_path}: {predicted} (confiança: {confidence:.2f})")
    else:
        print("Erro no treinamento!")
    
    print("\n" + "=" * 70)
    print("MELHORIAS IMPLEMENTADAS:")
    print("1. Data Augmentation: Rotação, flip, cor, brilho, contraste")
    print("2. Arquitetura CNN: BatchNorm, Dropout, múltiplas camadas")
    print("3. Otimização: AdamW, LR Scheduler, Gradient Clipping")
    print("4. Regularização: Label Smoothing, Weight Decay")
    print("5. Inicialização: Xavier/He para melhor convergência")
    print("6. Early Stopping: Evita overfitting")
    print("7. Pré-processamento: Filtros de nitidez e contraste")
    print("\nPara usar o modelo melhorado, atualize o main.py!")

if __name__ == "__main__":
    main()


