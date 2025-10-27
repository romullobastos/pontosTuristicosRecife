#!/usr/bin/env python3
"""
Jogo Educacional com Chatbot Multimodal
Desenvolvido do zero sem modelos pré-treinados
"""

import torch
import torch.nn as nn
from PIL import Image
import json
import os
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io

# Importar nossos módulos
from models.multimodal_model import EducationalMultimodalModel
from game.gamification import GamificationSystem
from game.photo_description_game import PhotoDescriptionGame
import sys
sys.path.append('training')
from improved_recife_trainer import ImprovedRecifeHistoricTrainer
from training.trainer import SimpleTokenizer, EducationalDataset
import torchvision.transforms as transforms

class EducationalGame:
    """
    Classe principal do jogo educacional
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando device: {self.device}")
        
        # Inicializar componentes
        self.tokenizer = SimpleTokenizer()
        self.gamification = GamificationSystem()
        self.photo_description_game = PhotoDescriptionGame()
        
        # Inicializar treinador MELHORADO de pontos históricos do Recife
        self.recife_trainer = ImprovedRecifeHistoricTrainer()
        
        # Tentar carregar modelo treinado
        if not self.recife_trainer.load_model():
            print("Modelo treinado nao encontrado!")
            print("Execute o treinamento melhorado primeiro: python train_improved_model.py")
        else:
            print("Modelo de pontos historicos do Recife carregado com sucesso!")
        
        # Usar apenas o modelo de pontos históricos (não precisamos do modelo multimodal antigo)
        self.model = None  # Não vamos usar o modelo multimodal antigo
        
        # Transformações para imagens
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Cache para melhorar consistência das respostas
        self.image_cache = {}
        
        print("Jogo inicializado com sucesso!")
    
    
    def process_image_and_question(self, image_data, question, player_id):
        """
        Processa uma imagem e pergunta do jogador
        """
        start_time = time.time()
        
        try:
            # Decodificar imagem
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Se image_data já é uma string base64, usar diretamente
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
            else:
                # Se já é bytes, usar diretamente
                image_bytes = image_data
            
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Criar nome único para arquivo temporário (sem hashlib)
            timestamp = str(int(time.time() * 1000))
            temp_path = f"temp_{timestamp}.jpg"
            image.save(temp_path)
            
            # Análise da imagem usando o treinador de pontos históricos
            predicted_class, confidence = self.recife_trainer.predict(temp_path)
            
            # Debug: imprimir predição
            print(f"[DEBUG] Predição: {predicted_class}, Confianca: {confidence:.2f}")
            
            # Obter informações do local
            location_info = self.recife_trainer.get_location_info(predicted_class)
            
            # Gerar resposta baseada na predição
            answer = location_info['nome']
            explanation = f"{location_info['descricao']}. {location_info['historia']}"
            
            # Debug: imprimir resposta
            print(f"[DEBUG] Resposta: {answer}")
            
            # Avaliar resposta
            score = self._evaluate_answer(answer, question)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Atualizar gamificação
            self.gamification.submit_answer(player_id, score > 0, response_time, "historic_location")
            
            result = {
                'success': True,
                'answer': answer,
                'explanation': explanation,
                'score': score,
                'player_stats': self.gamification.get_player_stats(player_id),
                'response_time': f"{response_time:.2f}s",
                'confidence': f"{confidence:.2f}",
                'location_info': location_info
            }
            
            # Cache desabilitado completamente
            # if image_hash:
            #     self.image_cache[image_hash] = result
            
            # Limpar arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            return {'success': False, 'error': f"Erro ao processar imagem: {e}"}
    
    def _analyze_image_features(self, image):
        """Simula análise de características visuais da imagem"""
        # Em produção, isso seria feito pelo modelo de visão
        # Por agora, simular baseado em características comuns
        import random
        
        # Simular detecção de características visuais
        features = {
            "has_stripes": random.random() < 0.3,  # 30% chance de ter listras
            "has_fur": random.random() < 0.7,     # 70% chance de ter pelo
            "has_wings": random.random() < 0.2,   # 20% chance de ter asas
            "has_trunk": random.random() < 0.1,  # 10% chance de ter tromba
            "has_mane": random.random() < 0.15,   # 15% chance de ter juba
            "is_green": random.random() < 0.4,    # 40% chance de ser verde
            "has_leaves": random.random() < 0.3, # 30% chance de ter folhas
            "is_mechanical": random.random() < 0.2 # 20% chance de ser mecânico
        }
        
        return features
    
    def _detect_zebra_features(self, image):
        """Detecta características específicas de zebra na imagem"""
        # Simular análise mais sofisticada para zebras
        import random
        
        # Converter imagem para análise básica
        import numpy as np
        img_array = np.array(image)
        
        # Simular detecção de listras (padrão preto e branco)
        # Em produção, isso seria feito com análise de gradientes
        has_stripes = random.random() < 0.3  # Reduzido para 30% - listras são raras
        
        # Simular detecção de formato de cavalo/equídeo
        is_horse_like = random.random() < 0.7  # 70% chance de ser formato de cavalo
        
        # Simular detecção de ambiente (savana/grama)
        has_grass_background = random.random() < 0.8  # 80% chance de ter grama ao fundo
        
        return {
            "has_stripes": has_stripes,
            "is_horse_like": is_horse_like,
            "has_grass_background": has_grass_background,
            "is_zebra": has_stripes and is_horse_like and has_grass_background
        }
    
    def _analyze_image_content(self, image):
        """Análise usando modelo treinado para pontos históricos do Recife"""
        try:
            # Salvar imagem temporariamente para análise
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            
            # Usar modelo treinado para predição
            predicted_class, confidence = self.recife_trainer.predict(temp_path)
            
            # Remover arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Retornar resultado da análise
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "model_trained": True,
                "location_info": self.recife_trainer.get_location_info(predicted_class)
            }
            
        except Exception as e:
            print(f"Erro na analise com modelo treinado: {e}")
            # Fallback para análise básica
            return self._analyze_image_content_fallback(image)
    
    def _analyze_image_content_fallback(self, image):
        """Análise básica como fallback"""
        import numpy as np
        
        # Converter imagem para análise
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Análise de cores dominantes
        colors = img_array.reshape(-1, 3)
        avg_r = np.mean(colors[:, 0])
        avg_g = np.mean(colors[:, 1])
        avg_b = np.mean(colors[:, 2])
        
        # Determinar cor dominante
        dominant_color = "unknown"
        if avg_r > 200 and avg_g > 200 and avg_b > 200:
            dominant_color = "white"
        elif avg_r < 50 and avg_g < 50 and avg_b < 50:
            dominant_color = "black"
        elif avg_g > avg_r + 30 and avg_g > avg_b + 30:
            dominant_color = "green"
        elif avg_b > avg_r + 30 and avg_b > avg_g + 30:
            dominant_color = "blue"
        elif avg_r > avg_g + 30 and avg_r > avg_b + 30:
            dominant_color = "red"
        elif avg_r > 150 and avg_g > 100 and avg_b < 100:
            dominant_color = "yellow"
        elif 80 < avg_r < 150 and 60 < avg_g < 120 and 40 < avg_b < 100:
            dominant_color = "brown"
        elif abs(avg_r - avg_g) < 20 and abs(avg_g - avg_b) < 20:
            dominant_color = "gray"
        
        return {
            "predicted_class": "objeto não identificado",
            "confidence": 0.1,
            "model_trained": False,
            "dominant_color": dominant_color
        }
    
    def _classify_image_content(self, features):
        """Classificação usando modelo treinado para pontos históricos do Recife"""
        
        # Se temos resultado do modelo treinado, usar ele
        if features.get("model_trained", False):
            predicted_class = features.get("predicted_class", "local não identificado")
            confidence = features.get("confidence", 0.0)
            location_info = features.get("location_info", {})
            
            print(f"Modelo treinado: {location_info.get('nome', predicted_class)} (confianca: {confidence:.2f})")
            
            # Se a confiança é alta, usar a predição
            if confidence > 0.3:
                return predicted_class
            else:
                print("Confianca baixa, usando fallback")
        
        # Fallback para análise básica
        dominant_color = features.get("dominant_color", "unknown")
        
        # Lógica simples baseada na cor dominante para pontos históricos
        if dominant_color == "brown":
            return "marco_zero"  # Marco Zero tem calçada de pedra marrom
        elif dominant_color == "yellow":
            return "praia_boa_viagem"  # Praia tem areia amarelada
        elif dominant_color == "gray":
            return "casa_da_cultura"  # Casa da Cultura é de pedra cinza
        elif dominant_color == "blue":
            return "praia_boa_viagem"  # Praia tem mar azul
        elif dominant_color == "green":
            return "praça_do_arsenal"  # Praças têm vegetação verde
        else:
            return "local não identificado"
    
    def _generate_answer(self, outputs, question, image_hash=None, image=None):
        """Gera resposta baseada nas saídas do modelo"""
        # Implementação melhorada com análise de palavras-chave
        answer_logits = outputs['answer_logits']
        _, predicted = torch.max(answer_logits, 1)
        
        # Analisar a pergunta para dar resposta mais específica
        question_lower = question.lower()
        
        # Se temos a imagem, fazer análise completa do conteúdo
        if image is not None:
            # Verificar cache primeiro
            if image_hash and image_hash in self.image_cache:
                cached_answer = self.image_cache[image_hash]
                return cached_answer
            
            # Análise completa da imagem
            features = self._analyze_image_content(image)
            classification = self._classify_image_content(features)
            
            # Debug: mostrar análise da imagem
            print(f"Analise da Imagem:")
            if features.get("model_trained", False):
                print(f"  Modelo treinado: {features.get('predicted_class', 'unknown')}")
                print(f"  Confianca: {features.get('confidence', 0.0):.2f}")
            else:
                print(f"  Cor dominante: {features.get('dominant_color', 'unknown')}")
                print(f"  Usando analise basica (fallback)")
            
            print(f"Classificacao final: {classification}")
            
            # Gerar resposta baseada na classificação
            if classification != "local não identificado":
                # Obter informações do local
                location_info = features.get("location_info", {})
                if location_info:
                    answer = f"Este é o {location_info.get('nome', classification)}"
                else:
                    answer = f"Este é o {classification.replace('_', ' ').title()}"
                # Salvar no cache
                if image_hash:
                    self.image_cache[image_hash] = answer
                return answer
        
        # Detectar tipo de pergunta sobre pontos históricos
        if "onde" in question_lower or "local" in question_lower or "histórico" in question_lower:
            # Verificar se já temos uma resposta em cache para esta imagem
            if image_hash and image_hash in self.image_cache:
                cached_answer = self.image_cache[image_hash]
                return cached_answer
            
            # Simular análise da imagem (em produção seria mais sofisticada)
            import random
            
            # Para perguntas sobre animais, dar respostas mais específicas
            # Cavalos são mais comuns que zebras, então maior probabilidade
            if "cavalo" in question_lower or random.random() < 0.6:  # 60% chance de identificar cavalo
                answer = "Este é um cavalo"
                # Salvar no cache
                if image_hash:
                    self.image_cache[image_hash] = answer
                return answer
            elif "zebra" in question_lower or random.random() < 0.2:  # 20% chance de identificar zebra
                answer = "Esta é uma zebra"
                # Salvar no cache
                if image_hash:
                    self.image_cache[image_hash] = answer
                return answer
            
            animals = [
                "Este é um cão",
                "Este é um gato", 
                "Esta é uma zebra",
                "Este é um leão",
                "Este é um elefante",
                "Este é um pássaro",
                "Este é um cavalo",
                "Esta é uma vaca"
            ]
            answer = random.choice(animals)
            # Salvar no cache
            if image_hash:
                self.image_cache[image_hash] = answer
            return answer
        
        elif "planta" in question_lower or "que tipo de planta" in question_lower:
            plants = [
                "Esta é uma flor",
                "Esta é uma árvore",
                "Esta é uma folha",
                "Esta é uma grama",
                "Esta é uma rosa",
                "Esta é uma palmeira"
            ]
            return random.choice(plants)
        
        elif "objeto" in question_lower or "que objeto" in question_lower:
            objects = [
                "Este é um carro",
                "Esta é uma casa",
                "Este é um livro",
                "Esta é uma mesa",
                "Este é um telefone",
                "Esta é uma bicicleta"
            ]
            return random.choice(objects)
        
        else:
            # Resposta genérica baseada na predição do modelo
            answers = {
                0: "Este é um animal",
                1: "Esta é uma planta", 
                2: "Este é um objeto",
                3: "Esta é uma paisagem"
            }
            return answers.get(predicted.item(), "Não consegui identificar com certeza.")
    
    def _generate_explanation(self, outputs, question, answer=None):
        """Gera explicação educativa sobre pontos históricos do Recife"""
        explanation_logits = outputs['explanation_logits']
        _, predicted = torch.max(explanation_logits, 1)
        
        # Se temos uma resposta específica, gerar explicação baseada nela
        if answer:
            answer_lower = answer.lower()
            
            # Explicações específicas para pontos históricos do Recife
            if "marco zero" in answer_lower:
                return "O Marco Zero é a praça principal do Recife Antigo, onde a cidade foi fundada em 1537. É um marco histórico que representa o início da colonização portuguesa no Nordeste. A praça tem calçada de pedra portuguesa e vista para o mar, sendo um dos pontos turísticos mais importantes da cidade."
            elif "praia boa viagem" in answer_lower:
                return "A Praia de Boa Viagem é a principal praia urbana do Recife, conhecida por sua extensa faixa de areia e calçadão. É uma das praias mais famosas do Nordeste, com coqueiros e vista para o mar. O nome 'Boa Viagem' vem de uma capela construída no local para abençoar os navegantes."
            elif "casa da cultura" in answer_lower:
                return "A Casa da Cultura é um importante centro cultural do Recife, localizado no antigo prédio da Casa de Detenção, construído em 1855. Foi transformada em centro cultural em 1976 e hoje abriga lojas de artesanato, onde as antigas celas foram transformadas em espaços comerciais. É um exemplo de preservação do patrimônio histórico."
            elif "forte das cinco pontas" in answer_lower:
                return "O Forte das Cinco Pontas é uma fortaleza histórica construída pelos holandeses em 1630. Tem formato pentagonal único e foi construído para defender a cidade. Hoje abriga o Museu da Cidade do Recife e é um importante marco da arquitetura militar colonial."
            elif "igreja são pedro dos clérigos" in answer_lower:
                return "A Igreja de São Pedro dos Clérigos é uma igreja barroca do século XVIII, construída entre 1728 e 1782. É conhecida por sua torre alta e fachada barroca elaborada. O interior é ricamente decorado com dourado e representa um dos melhores exemplos da arquitetura religiosa colonial do Recife."
            elif "rua do bom jesus" in answer_lower:
                return "A Rua do Bom Jesus é uma das ruas mais antigas do Recife, localizada no bairro do Recife Antigo. É conhecida por suas casas coloridas e arquitetura colonial. A rua tem importância histórica e cultural, sendo um dos principais pontos turísticos da cidade."
            elif "ponte mauricio de nassau" in answer_lower:
                return "A Ponte Mauricio de Nassau é uma ponte histórica que conecta o Recife Antigo ao centro da cidade. Foi construída durante o período holandês e é uma das pontes mais antigas do Brasil. É um importante marco da engenharia colonial e da história da cidade."
            elif "praça do arsenal" in answer_lower:
                return "A Praça do Arsenal é uma praça histórica localizada no Recife Antigo, próxima ao Marco Zero. É um espaço público importante que preserva a arquitetura colonial e oferece vista para o mar. A praça é cercada por prédios históricos e é um local de encontro e lazer."
            elif "igreja nossa senhora do carmo" in answer_lower:
                return "A Igreja de Nossa Senhora do Carmo é uma igreja histórica do Recife, conhecida por sua arquitetura barroca e interior ricamente decorado. É um importante marco religioso e arquitetônico da cidade, representando a tradição católica colonial."
            elif "palácio da justiça" in answer_lower:
                return "O Palácio da Justiça é um prédio histórico que abriga o Tribunal de Justiça de Pernambuco. É um exemplo da arquitetura neoclássica e representa a importância da justiça na história da cidade. O prédio é um marco da arquitetura institucional do Recife."
            
            # Explicação genérica para outros locais históricos
            else:
                return f"Este é um importante ponto histórico do Recife. A cidade tem uma rica história colonial, com influências portuguesas e holandesas. Cada local histórico conta uma parte da história da cidade e do Brasil."
        
        # Explicação genérica baseada na pergunta
        question_lower = question.lower()
        if "onde" in question_lower:
            return "Esta pergunta envolve localização geográfica. O sistema analisou elementos visuais que podem indicar características arquitetônicas e geográficas específicas do Recife."
        elif "histórico" in question_lower or "historia" in question_lower:
            return "Esta pergunta envolve conhecimento histórico. O sistema examinou características arquitetônicas e visuais para identificar o contexto histórico do local."
        elif "que" in question_lower:
            return "Esta é uma pergunta sobre identificação de locais históricos. O sistema analisou as características visuais da imagem para identificar o ponto histórico específico do Recife."
        elif "como" in question_lower:
            return "Esta pergunta envolve análise de características arquitetônicas e históricas. O sistema examinou os detalhes visuais para entender melhor o local histórico."
        else:
            return "Esta é uma pergunta interessante sobre pontos históricos do Recife. O sistema examinou cuidadosamente os detalhes da imagem para identificar o local histórico específico e fornecer informações educativas sobre sua importância."
    
    def _evaluate_answer(self, answer, question):
        """Avalia a qualidade da resposta"""
        # Implementação simples de avaliação
        score = 0
        
        # Verificar se a resposta contém informações relevantes
        if answer and len(answer) > 10:
            score += 1
        
        # Verificar se a resposta é específica sobre pontos históricos
        historic_keywords = [
            'marco zero', 'praia boa viagem', 'casa da cultura', 'forte das cinco pontas',
            'igreja são pedro dos clérigos', 'rua do bom jesus', 'ponte mauricio de nassau',
            'praça do arsenal', 'igreja nossa senhora do carmo', 'palácio da justiça',
            'recife', 'histórico', 'história', 'colonial', 'português', 'holandês'
        ]
        
        answer_lower = answer.lower()
        for keyword in historic_keywords:
            if keyword in answer_lower:
                score += 1
                break
        
        return min(score, 5)  # Máximo de 5 pontos

# Configuração da aplicação Flask
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Instanciar o jogo
game = EducationalGame()

@app.route('/')
def index():
    """Página principal do jogo"""
    return render_template('index.html')

@app.route('/api/process_image', methods=['POST'])
def process_image():
    """Processa imagem e pergunta do usuário"""
    try:
        data = request.json
        image_data = data.get('image')
        question = data.get('question', 'Que local histórico é este?')
        player_id = data.get('player_id', 'Jogador')
        
        if not image_data:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400
        
        # Decodificar imagem base64
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Processar com o jogo
        result = game.process_image_and_question(image_data, question, player_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_image_legacy():
    """Processa imagem e pergunta do usuário (rota legada)"""
    try:
        data = request.json
        image_data = data.get('image')
        question = data.get('question', 'Que local histórico é este?')
        player_name = data.get('player_name', 'Jogador')
        
        if not image_data:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400
        
        # Decodificar imagem base64
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Processar com o jogo
        result = game.process_image_and_question(image_data, question, player_name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Retorna o ranking de jogadores"""
    return jsonify(game.gamification.get_leaderboard())

@app.route('/api/create_player', methods=['POST'])
def create_player():
    """Cria um novo jogador"""
    try:
        data = request.json
        player_name = data.get('name', 'Jogador')
        
        # Gerar ID único para o jogador
        import uuid
        player_id = str(uuid.uuid4())
        
        # Criar jogador no sistema de gamificação
        player = game.gamification.create_player(player_id, player_name)
        
        return jsonify({
            'success': True,
            'player_id': player_id,
            'message': f'Jogador {player_name} criado com sucesso!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player_stats/<player_id>', methods=['GET'])
def get_player_stats_by_id(player_id):
    """Retorna estatísticas do jogador por ID"""
    try:
        # Buscar jogador por ID
        player = game.gamification.players.get(player_id)
        if not player:
            return jsonify({'error': 'Jogador não encontrado'}), 404
        
        stats = {
            'name': player.name,
            'level': player.level,
            'experience': player.experience,
            'coins': player.coins,
            'streak': player.streak,
            'total_correct': player.total_correct,
            'total_attempts': player.total_attempts,
            'achievements': player.achievements
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player_stats', methods=['GET'])
def get_player_stats():
    """Retorna estatísticas do jogador"""
    player_name = request.args.get('player_name', 'Jogador')
    return jsonify(game.gamification.get_player_stats(player_name))

# Rotas para o jogo de descrições de fotos
@app.route('/api/photo_game/random_photo', methods=['GET'])
def get_random_photo():
    """Retorna uma foto aleatória para o jogo de descrições"""
    try:
        photo = game.photo_description_game.get_random_photo()
        if photo:
            # Ler imagem e converter para base64
            import base64
            image_path = photo['image_path']
            
            try:
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    photo['image_data'] = f"data:image/jpeg;base64,{img_base64}"
            except Exception as e:
                print(f"Erro ao carregar imagem: {e}")
                photo['image_data'] = None
            
            return jsonify({
                'success': True,
                'photo': photo
            })
        else:
            return jsonify({'error': 'Nenhuma foto disponível'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/photo_game/submit_description', methods=['POST'])
def submit_description():
    """Submete descrição do usuário para avaliação"""
    try:
        data = request.json
        description = data.get('description', '')
        
        if not description.strip():
            return jsonify({'error': 'Descrição não pode estar vazia'}), 400
        
        result = game.photo_description_game.submit_description(description)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/photo_game/stats', methods=['GET'])
def get_photo_game_stats():
    """Retorna estatísticas do jogo de descrições"""
    try:
        stats = game.photo_description_game.get_game_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/photo_game/reset', methods=['POST'])
def reset_photo_game():
    """Reseta o jogo de descrições"""
    try:
        game.photo_description_game.reset_game()
        return jsonify({
            'success': True,
            'message': 'Jogo resetado com sucesso'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve arquivos estáticos incluindo imagens"""
    return app.send_static_file(filename)

if __name__ == '__main__':
    print("Iniciando servidor do Jogo Educacional de Pontos Historicos do Recife...")
    print("Acesse: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
