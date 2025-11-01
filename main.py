#!/usr/bin/env python3
"""
Jogo Educacional com Chatbot Multimodal
Desenvolvido do zero sem modelos pré-treinados
"""

import torch
from PIL import Image
import json
import os
import time
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import base64
import io
import mimetypes
import re

# Importar nossos módulos
from game.gamification import GamificationSystem
from game.photo_description_game import PhotoDescriptionGame
import sys
sys.path.append('training')
from improved_recife_trainer import ImprovedRecifeHistoricTrainer

class EducationalGame:
    """
    Classe principal do jogo educacional
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando device: {self.device}")
        
        # Inicializar componentes
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
            
            # Limpar arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            return {'success': False, 'error': f"Erro ao processar imagem: {e}"}
    
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
app.secret_key = 'dev-secret-key'  # apenas para desenvolvimento

# ---- Auth storage helpers ----
import json, os, uuid, datetime
USERS_PATH = os.path.join('data', 'users.json')

def _load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    try:
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: dict):
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def _ensure_admin_seed():
    users = _load_users()
    username = 'admin'
    desired_email = 'email@email.com'
    desired_password = 'senhadoadmin'
    key_map = {k.lower(): k for k in users.keys()}
    if 'admin' in key_map:
        k = key_map['admin']
        # atualizar email/senha e flags
        users[k]['email'] = desired_email
        users[k]['password_hash'] = generate_password_hash(desired_password)
        users[k]['is_admin'] = True
        if not users[k].get('player_id'):
            users[k]['player_id'] = str(uuid.uuid4())
        # garantir lista de player_ids
        if not isinstance(users[k].get('player_ids'), list):
            users[k]['player_ids'] = []
        if users[k]['player_id'] not in users[k]['player_ids']:
            users[k]['player_ids'].append(users[k]['player_id'])
        # garantir player associado
        if users[k]['player_id'] not in game.gamification.players:
            game.gamification.create_player(users[k]['player_id'], username)
    else:
        users[username] = {
            'email': desired_email,
            'password_hash': generate_password_hash(desired_password),
            'is_admin': True,
            'player_id': str(uuid.uuid4()),
            'player_ids': [],
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        game.gamification.create_player(users[username]['player_id'], username)
        users[username]['player_ids'].append(users[username]['player_id'])
    try:
        game.gamification.save_to_file('data/players.json')
    except Exception:
        pass
    _save_users(users)

# ---- Auth endpoints ----
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form['password']
        users = _load_users()
        # mapear nome informado para chave real (case-insensitive)
        key_map = {k.lower(): k for k in users.keys()}
        real_key = key_map.get(username.lower())
        if real_key and check_password_hash(users[real_key]['password_hash'], password):
            session['username'] = real_key
            return redirect(url_for('index'))
        # inválido: retorna com erro
        return render_template('login.html', error='Usuário ou senha inválidos')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email = request.form['email']
        password = request.form['password']
        users = _load_users()
        # bloquear duplicatas ignorando maiúsculas/minúsculas
        if username.lower() not in {k.lower(): k for k in users.keys()}:
            users[username] = {
                'email': email,
                'password_hash': generate_password_hash(password),
                'is_admin': False,
                'player_id': str(uuid.uuid4()),
                'created_at': datetime.datetime.utcnow().isoformat()
            }
            # garantir player associado
            game.gamification.create_player(users[username]['player_id'], username)
            try:
                game.gamification.save_to_file('data/players.json')
            except Exception:
                pass
            _save_users(users)
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        new_password = request.form.get('new_password', '')
        users = _load_users()
        if username in users:
            u = users[username]
            # valida par username+email
            if u.get('email', '').strip().lower() == email.lower() and new_password:
                users[username]['password_hash'] = generate_password_hash(new_password)
                _save_users(users)
                return redirect(url_for('login'))
    return render_template('reset.html')

@app.route('/me', methods=['GET'])
def me():
    if 'username' in session:
        users = _load_users()
        u = users.get(session['username'])
        if u:
            return jsonify({'username': session['username'], 'is_admin': bool(u.get('is_admin')), 'player_id': u.get('player_id'), 'player_ids': u.get('player_ids', [])})
        return jsonify({'username': session['username'], 'is_admin': False})
    return jsonify({}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({})

# Instanciar o jogo e garantir admin seed
game = EducationalGame()
_ensure_admin_seed()

@app.route('/')
def index():
    """Página principal do jogo"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    """Interface de chatbot interativo"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/api/process_image', methods=['POST'])
def process_image():
    """Processa imagem e pergunta do usuário"""
    try:
        data = request.json
        image_data = data.get('image')
        question = data.get('question', 'Que local histórico é este?')
        player_id = data.get('player_id', 'Jogador')
        no_xp = bool(data.get('no_xp', False))
        
        if not image_data:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400
        
        # Decodificar imagem base64
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        if no_xp:
            # Caminho leve: apenas identificar e responder sem alterar XP
            temp_path = 'temp_analysis_image.jpg'
            image.save(temp_path)
            try:
                predicted_class, confidence = game.recife_trainer.predict(temp_path)
                info = game.recife_trainer.get_location_info(predicted_class)
                answer = f"Este é o {info.get('nome', predicted_class.replace('_',' ').title())}"
                explanation = info.get('descricao', '')
                return jsonify({
                    'success': True,
                    'answer': answer,
                    'explanation': explanation,
                    'response_time': 'Rápido',
                    'confidence': f"{confidence:.2f}",
                })
            finally:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
        else:
            # Processar com o jogo (pode alterar XP)
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

# --------- PHOTO GAME ENDPOINTS ---------

def _encode_image_as_data_url(image_path: str) -> str:
    try:
        if not image_path:
            return ''
        # Caminho relativo ao projeto
        path = image_path if os.path.exists(image_path) else os.path.join(os.getcwd(), image_path)
        with open(path, 'rb') as f:
            b = f.read()
        mime, _ = mimetypes.guess_type(path)
        mime = mime or 'image/jpeg'
        b64 = base64.b64encode(b).decode('utf-8')
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ''

@app.route('/api/photo_game/random_photo', methods=['GET'])
def api_random_photo():
    try:
        photo = game.photo_description_game.get_random_photo()
        if not photo:
            return jsonify({'success': False, 'error': 'Sem fotos no dataset'}), 404
        # Anexar image_data e hints (se disponíveis no dataset completo)
        full = game.photo_description_game.get_photo_by_id(photo['id']) or {}
        image_data = _encode_image_as_data_url(full.get('image_path') or photo.get('image_path'))
        photo_out = {
            **photo,
            'image_data': image_data,
            'hints': full.get('hints') or {}
        }
        return jsonify({'success': True, 'photo': photo_out})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/photo_game/get_photo/<photo_id>', methods=['GET'])
def api_get_photo(photo_id):
    try:
        full = game.photo_description_game.get_photo_by_id(photo_id)
        if not full:
            return jsonify({'success': False, 'error': 'Foto não encontrada'}), 404
        image_data = _encode_image_as_data_url(full.get('image_path'))
        return jsonify({'success': True, 'photo': {**full, 'image_data': image_data}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/photo_game/submit_description', methods=['POST'])
def api_submit_description():
    try:
        data = request.json or {}
        description = data.get('description', '')
        photo_id = data.get('photo_id')
        # Permitir validação sem photo_id (legado), mas ideal setar
        if photo_id:
            game.photo_description_game.set_current_photo(photo_id)
        result = game.photo_description_game.submit_description(description)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/photo_game/mystery_guess', methods=['POST'])
def api_mystery_guess():
    """Valida palpite do Modo Mistério usando o mecanismo de NLP do submit_description."""
    try:
        start_t = time.time()
        data = request.json or {}
        guess = data.get('guess', '')
        photo_id = data.get('photo_id')
        player_id = data.get('player_id')
        hints_used = int(data.get('hints_used', 0) or 0)
        # Resolver player_id a partir da sessão se não vier no payload
        if not player_id and 'username' in session:
            try:
                users = _load_users()
                u = users.get(session['username'])
                if u and u.get('player_id'):
                    player_id = u.get('player_id')
            except Exception:
                pass
        if not photo_id:
            return jsonify({'success': False, 'error': 'photo_id é obrigatório'}), 400
        # posicionar foto atual
        if not game.photo_description_game.set_current_photo(photo_id):
            return jsonify({'success': False, 'error': 'Foto não encontrada'}), 404
        # avaliar via NLP existente
        # --- Balanced evaluation: token rule + dynamic NLP threshold ---
        def _normalize(txt: str) -> str:
            if not isinstance(txt, str):
                return ''
            t = txt.lower()
            t = re.sub(r'[\u0300-\u036f]', '', re.sub(r'[^\w\s]', ' ', t, flags=re.UNICODE))
            t = re.sub(r'\s+', ' ', t).strip()
            return t
        def _strip_generics(t: str) -> str:
            # remove tipos e artigos/preposições comuns
            stop = r"\b(da|de|do|das|dos|del|di|du|the|of|and|e)\b"
            kinds = r"\b(igreja|bas[\w]*lica|palacio|rua|mercado|forte|teatro|concatedral|museu)\b"
            t = re.sub(stop, ' ', t)
            t = re.sub(kinds, ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        photo = game.photo_description_game.get_photo_by_id(photo_id) or {}
        name_norm = _strip_generics(_normalize(photo.get('name', '')))
        guess_norm = _strip_generics(_normalize(guess))
        name_tokens = [tok for tok in set(name_norm.split(' ')) if len(tok) >= 3]
        guess_tokens = [tok for tok in set(guess_norm.split(' ')) if len(tok) >= 3]
        common = set(name_tokens).intersection(set(guess_tokens))
        # regra de tokens: nomes curtos (<=1 token) precisam 1 comum; compostos precisam >=2
        tokens_needed = 1 if len(name_tokens) <= 1 else 2
        token_ok = len(common) >= tokens_needed

        # NLP score
        eval_result = game.photo_description_game.submit_description(guess)
        final_score = float(eval_result.get('final_score', 0.0))
        # Threshold dinâmico baseado em dicas usadas (até 2 dicas reduzem um pouco)
        base_threshold = 0.45
        reduce = min(max(hints_used, 0), 2) * 0.03
        mystery_threshold = max(0.35, base_threshold - reduce)
        is_correct = token_ok or (final_score >= mystery_threshold)
        # atualizar gamificação/XP geral
        points_earned = 0  # será atualizado com o valor real do submit_answer
        try:
            response_time = time.time() - start_t
            if player_id:
                # garantir que o jogador exista na gamificação
                player = game.gamification.get_player(player_id)
                if player is None:
                    # tentar nome amigável a partir do usuário logado
                    pname = session['username'] if 'username' in session else 'Jogador'
                    game.gamification.create_player(player_id, pname)
                
                # Log XP antes
                xp_before = player.experience if player else 0
                print(f"[MYSTERY MODE DEBUG] XP antes: {xp_before}")
                
                # registra tentativa no sistema de gamificação (categoria: mystery_mode)
                # submit_answer calcula e adiciona XP automaticamente (base + velocidade + streak - penalidade de dicas)
                result = game.gamification.submit_answer(player_id, bool(is_correct), response_time, "mystery_mode", hints_used)
                
                # Usar o XP real calculado pelo submit_answer para exibir no frontend
                points_earned = result.get('points', 0)
                
                # Verificar novas conquistas desbloqueadas
                new_achievements = result.get('new_achievements', [])
                
                # Log XP depois
                player = game.gamification.get_player(player_id)
                xp_after = player.experience if player else 0
                print(f"[MYSTERY MODE DEBUG] XP depois: {xp_after}, XP adicionado (real): {points_earned}, Diferença: {xp_after - xp_before}")
                
                # persistir progresso
                try:
                    game.gamification.save_to_file('data/players.json')
                except Exception as e:
                    print(f"[MYSTERY MODE DEBUG] Erro ao salvar: {e}")
        except Exception as e:
            print(f"[MYSTERY MODE DEBUG] Erro geral: {e}")
            import traceback
            traceback.print_exc()
            new_achievements = []  # Inicializar em caso de erro
        
        # Inicializar variável de novas conquistas se não foi definida
        if 'new_achievements' not in locals():
            new_achievements = []
        
        # incluir estatísticas atualizadas do jogador (se houver)
        updated_stats = None
        try:
            if player_id:
                updated_stats = game.gamification.get_player_stats(player_id)
        except Exception:
            updated_stats = None
        
        # Preparar resposta com novas conquistas se houver
        response_data = {
            'success': True,
            'is_correct': is_correct,
            'final_score': final_score,
            'points_earned': points_earned,
            'photo_name': photo.get('name', ''),
            'player_stats': updated_stats
        }
        
        # Adicionar novas conquistas se foram desbloqueadas
        if new_achievements:
            response_data['new_achievements'] = [
                {
                    'id': ach.id,
                    'name': ach.name,
                    'description': ach.description,
                    'icon': ach.icon
                } for ach in new_achievements
            ]
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Admin: resetar jogadores/ranking em runtime
@app.route('/api/admin/reset_players', methods=['POST'])
def admin_reset_players():
    try:
        game.gamification.reset_players('data/players.json')
        return jsonify({'success': True, 'message': 'Jogadores resetados'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Admin: recarregar jogadores do arquivo
@app.route('/api/admin/reload_players', methods=['POST'])
def admin_reload_players():
    try:
        ok = game.gamification.load_from_file('data/players.json')
        return jsonify({'success': bool(ok)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        # Salvar estado imediatamente
        try:
            game.gamification.save_to_file('data/players.json')
        except Exception:
            pass

        # Se há usuário logado, associar este novo player à conta
        try:
            if 'username' in session:
                users = _load_users()
                u = users.get(session['username'])
                if u is not None:
                    if not isinstance(u.get('player_ids'), list):
                        u['player_ids'] = []
                    if player_id not in u['player_ids']:
                        u['player_ids'].append(player_id)
                    # manter player_id principal como o mais recente se desejar
                    users[session['username']] = u
                    _save_users(users)
        except Exception:
            pass

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
        
        # Obter detalhes das conquistas
        achievements_details = []
        for achievement_id in player.achievements:
            for achievement in game.gamification.achievements:
                if achievement.id == achievement_id:
                    achievements_details.append({
                        'id': achievement.id,
                        'name': achievement.name,
                        'description': achievement.description,
                        'icon': achievement.icon
                    })
                    break
        
        # Obter atividades recentes
        recent_activities = game.gamification.get_recent_activities(player_id, limit=5)
        
        stats = {
            'name': player.name,
            'level': player.level,
            'experience': player.experience,
            'streak': player.streak,
            'total_correct': player.total_correct,
            'total_attempts': player.total_attempts,
            'achievements': player.achievements,  # IDs para compatibilidade
            'achievements_details': achievements_details,  # Detalhes completos
            'recent_activities': recent_activities  # Atividades recentes
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player_stats', methods=['GET'])
def get_player_stats():
    """Retorna estatísticas do jogador"""
    player_name = request.args.get('player_name', 'Jogador')
    return jsonify(game.gamification.get_player_stats(player_name))

@app.route('/api/compare_visual_similarity', methods=['POST'])
def compare_visual_similarity():
    """
    Compara visualmente duas imagens usando deep learning
    Retorna uma similaridade entre 0 e 1 e adiciona pontos ao XP
    """
    try:
        data = request.json
        user_image = data.get('user_image')  # Base64 da foto do usuário
        target_location = data.get('target_location')  # ID da foto do desafio
        player_id = data.get('player_id', 'default_player')
        
        if not user_image or not target_location:
            return jsonify({'error': 'Dados incompletos'}), 400
        
        # Decodificar imagem do usuário
        if isinstance(user_image, str) and user_image.startswith('data:image'):
            user_image = user_image.split(',')[1]
        
        user_image_bytes = base64.b64decode(user_image)
        user_img = Image.open(io.BytesIO(user_image_bytes)).convert('RGB')
        
        # Obter foto do desafio
        photo = game.photo_description_game.get_photo_by_id(target_location)
        if not photo:
            return jsonify({'error': 'Foto do desafio não encontrada'}), 404
        
        # Carregar foto do desafio
        target_img = Image.open(photo['image_path']).convert('RGB')
        
        # Calcular similaridade usando o modelo de pontos históricos
        similarity = game.recife_trainer.compare_images(user_img, target_img)
        
        # Calcular pontos baseado na similaridade
        # Similaridade de 0.8+ = excelente, 0.6+ = bom, 0.4+ = aceitável, 0.30+ = tentativa
        if similarity >= 0.8:
            points = photo['points']  # Pontos completos
        elif similarity >= 0.6:
            points = int(photo['points'] * 0.7)  # 70% dos pontos
        elif similarity >= 0.4:
            points = int(photo['points'] * 0.5)  # 50% dos pontos
        elif similarity >= 0.3:
            points = int(photo['points'] * 0.3)  # 30% dos pontos
        else:
            points = 0  # Abaixo de 30% não gera pontos
        
        # Adicionar pontos ao XP do jogador
        if player_id in game.gamification.players:
            player = game.gamification.players[player_id]
            player.experience += points
            player.total_attempts += 1
            
            # Se similaridade alta, considerar acerto
            if similarity >= 0.6:
                player.total_correct += 1
                player.streak += 1
            else:
                player.streak = 0
            
            # Verificar level up
            from game.gamification import GamificationSystem
            new_level = game.gamification._calculate_level(player.experience)
            level_up = new_level > player.level
            if level_up:
                player.level = new_level
            # Persistir alteração
            try:
                game.gamification.save_to_file('data/players.json')
            except Exception:
                pass
        
        return jsonify({
            'success': True,
            'similarity_score': float(similarity),
            'points_earned': points,
            'processing_time': 'Rápido'
        })
        
    except Exception as e:
        print(f"Erro ao comparar imagens: {e}")
        import traceback
        traceback.print_exc()
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

# Rota para Chrome DevTools (evitar 404 nos logs)
@app.route('/.well-known/appspecific/com.chrome.devtools.json', methods=['GET'])
def chrome_devtools():
    """Endpoint para Chrome DevTools - retorna vazio para evitar 404"""
    return '', 204  # 204 No Content

if __name__ == '__main__':
    print("Iniciando servidor do Jogo Educacional de Pontos Historicos do Recife...")
    print("Acesse: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
