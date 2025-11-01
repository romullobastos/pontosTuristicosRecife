import json
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

@dataclass
class Player:
    """Classe para representar um jogador"""
    id: str
    name: str
    level: int = 1
    experience: int = 0
    streak: int = 0
    total_correct: int = 0
    total_attempts: int = 0
    achievements: List[str] = None
    daily_challenges: Dict = None
    historic_locations_identified: int = 0  # Para conquista history_expert
    fastest_time: float = 999.0  # Para conquista speed_demon (inicia com valor alto)
    recent_activities: List[Dict] = None  # Lista de atividades recentes
    
    def __post_init__(self):
        if self.achievements is None:
            self.achievements = []
        if self.daily_challenges is None:
            self.daily_challenges = {}
        if not hasattr(self, 'historic_locations_identified'):
            self.historic_locations_identified = 0
        if not hasattr(self, 'fastest_time') or self.fastest_time == 0:
            self.fastest_time = 999.0
        if self.recent_activities is None:
            self.recent_activities = []

@dataclass
class Achievement:
    """Classe para representar uma conquista"""
    id: str
    name: str
    description: str
    condition: str
    reward_xp: int
    icon: str

class GamificationSystem:
    """
    Sistema de gamificação para o jogo educacional
    """
    
    def __init__(self):
        self.players: Dict[str, Player] = {}
        self.achievements = self._load_achievements()
        self.daily_challenges = self._generate_daily_challenges()
        # Persistência simples em JSON
        try:
            self.load_from_file('data/players.json')
        except Exception:
            pass
        
    def _load_achievements(self) -> List[Achievement]:
        """Carrega as conquistas disponíveis"""
        return [
            Achievement(
                id="first_image",
                name="Primeiro Passo",
                description="Analise sua primeira imagem",
                condition="total_attempts >= 1",
                reward_xp=50,
                icon="🌟"
            ),
            Achievement(
                id="detective_novice",
                name="Detetive Novato",
                description="Analise 10 imagens",
                condition="total_attempts >= 10",
                reward_xp=200,
                icon="🔍"
            ),
            Achievement(
                id="perfect_streak",
                name="Sequência Perfeita",
                description="Acertou 5 respostas seguidas",
                condition="streak >= 5",
                reward_xp=300,
                icon="🔥"
            ),
            Achievement(
                id="history_expert",
                name="Especialista em História",
                description="Identificou 20 locais históricos diferentes",
                condition="historic_locations_identified >= 20",
                reward_xp=500,
                icon="🏛️"
            ),
            Achievement(
                id="speed_demon",
                name="Demônio da Velocidade",
                description="Respondeu em menos de 10 segundos",
                condition="fastest_time <= 10",
                reward_xp=250,
                icon="⚡"
            ),
            Achievement(
                id="master_scientist",
                name="Cientista Mestre",
                description="Alcançou nível 10",
                condition="level >= 10",
                reward_xp=1000,
                icon="🧬"
            )
        ]
    
    def _generate_daily_challenges(self) -> List[Dict]:
        """Gera desafios diários"""
        return [
            {
                "id": "daily_images",
                "name": "Explorador Diário",
                "description": "Analise 5 imagens hoje",
                "target": 5,
                "reward_xp": 100,
                "progress": 0
            },
            {
                "id": "daily_accuracy",
                "name": "Precisão Perfeita",
                "description": "Mantenha 80% de precisão em 10 tentativas",
                "target": 8,
                "reward_xp": 200,
                "progress": 0
            },
            {
                "id": "daily_categories",
                "name": "Explorador Diverso",
                "description": "Explore 3 categorias diferentes",
                "target": 3,
                "reward_xp": 150,
                "progress": 0
            }
        ]
    
    def create_player(self, player_id: str, name: str) -> Player:
        """Cria um novo jogador"""
        player = Player(id=player_id, name=name)
        self.players[player_id] = player
        return player

    # ---- Persistência ----
    def save_to_file(self, file_path: str):
        import os, json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {
            pid: {
                'id': p.id,
                'name': p.name,
                'level': p.level,
                'experience': p.experience,
                'streak': p.streak,
                'total_correct': p.total_correct,
                'total_attempts': p.total_attempts,
                'achievements': p.achievements,
                'daily_challenges': p.daily_challenges,
                'historic_locations_identified': getattr(p, 'historic_locations_identified', 0),
                'fastest_time': getattr(p, 'fastest_time', 999.0),
                'recent_activities': getattr(p, 'recent_activities', []),
            } for pid, p in self.players.items()
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, file_path: str):
        import os, json
        if not os.path.exists(file_path):
            return False
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.players = {}
        for pid, obj in data.items():
            # Garantir que experience seja sempre inteiro (corrigir valores corrompidos)
            raw_xp = obj.get('experience', 0)
            experience = int(raw_xp) if isinstance(raw_xp, (int, float)) else 0
            player = Player(
                id=obj.get('id', pid),
                name=obj.get('name', 'Jogador'),
                level=obj.get('level', 1),
                experience=experience,
                streak=obj.get('streak', 0),
                total_correct=obj.get('total_correct', 0),
                total_attempts=obj.get('total_attempts', 0),
                achievements=obj.get('achievements', []),
                daily_challenges=obj.get('daily_challenges', {}),
                historic_locations_identified=obj.get('historic_locations_identified', 0),
                fastest_time=obj.get('fastest_time', 999.0),
                recent_activities=obj.get('recent_activities', [])
            )
            self.players[pid] = player
            # Verificar conquistas retroativamente ao carregar
            self._check_achievements_on_load(player)
        return True
    
    def _check_achievements_on_load(self, player: Player):
        """Verifica conquistas ao carregar jogador (sem adicionar XP novamente, apenas para garantir que estão na lista)"""
        # Garantir que atributos existam
        if not hasattr(player, 'historic_locations_identified'):
            player.historic_locations_identified = 0
        if not hasattr(player, 'fastest_time') or player.fastest_time == 0:
            player.fastest_time = 999.0
        
        # Se historic_locations_identified não foi salvo, usar total_correct como aproximação
        if player.historic_locations_identified == 0 and player.total_correct > 0:
            player.historic_locations_identified = player.total_correct
        
        for achievement in self.achievements:
            if achievement.id in player.achievements:
                continue
            
            # Verificar condição da conquista
            if self._evaluate_condition(achievement.condition, player):
                # Adicionar à lista de conquistas mas NÃO adicionar XP
                # (XP já foi dado quando desbloqueou originalmente, ou será dado na próxima ação)
                player.achievements.append(achievement.id)
                print(f"[ACHIEVEMENT DEBUG] Conquista {achievement.name} adicionada retroativamente ao jogador {player.name}")

    def reset_players(self, file_path: str = 'data/players.json'):
        """Remove todos os jogadores da memória e persiste arquivo vazio."""
        self.players = {}
        try:
            self.save_to_file(file_path)
        except Exception:
            pass
    
    def get_player(self, player_id: str) -> Optional[Player]:
        """Retorna um jogador pelo ID"""
        return self.players.get(player_id)
    
    def submit_answer(self, player_id: str, is_correct: bool, 
                     response_time: float, category: str = "general", hints_used: int = 0) -> Dict:
        """Registra uma resposta do jogador e atualiza estatísticas
        
        Args:
            player_id: ID do jogador
            is_correct: Se a resposta está correta
            response_time: Tempo de resposta em segundos
            category: Categoria da pergunta
            hints_used: Número de dicas usadas (cada dica reduz 5 XP)
        """
        print(f"[GAMIFICATION DEBUG] submit_answer chamado para player_id={player_id}, is_correct={is_correct}, category={category}, hints_used={hints_used}")
        player = self.get_player(player_id)
        if not player:
            return {"error": "Jogador não encontrado"}
        
        xp_before = player.experience
        print(f"[GAMIFICATION DEBUG] XP antes de submit_answer: {xp_before}")
        
        # Atualizar estatísticas
        player.total_attempts += 1
        if is_correct:
            player.total_correct += 1
            player.streak += 1
            
            # Rastrear tempo mais rápido (para conquista speed_demon)
            if response_time < player.fastest_time:
                player.fastest_time = response_time
            
            # Rastrear locais históricos identificados (para conquista history_expert)
            # Assumir que cada categoria "historic_location" ou "mystery_mode" identifica um local
            if category in ["historic_location", "mystery_mode"]:
                # Incrementar apenas uma vez por categoria (simplificado)
                # Em uma versão mais avançada, poderia rastrear locais específicos
                player.historic_locations_identified = max(player.historic_locations_identified, player.total_correct)
            
            # Calcular pontos (agora só XP, sem coins)
            base_points = 10
            speed_bonus = int(max(0, 20 - response_time))  # Bônus por velocidade (inteiro)
            streak_bonus = min(player.streak * 2, 20)  # Bônus por sequência
            hint_penalty = hints_used * 5  # Penalidade de 5 XP por dica usada
            
            points = int(max(5, base_points + speed_bonus + streak_bonus - hint_penalty))  # Mínimo 5 XP
            print(f"[GAMIFICATION DEBUG] Pontos calculados: base={base_points}, speed={speed_bonus}, streak={streak_bonus}, hint_penalty={hint_penalty}, total={points}")
            player.experience += points
            print(f"[GAMIFICATION DEBUG] XP depois de adicionar: {player.experience}")
            
        else:
            player.streak = 0
        
        # Verificar level up
        new_level = self._calculate_level(player.experience)
        level_up = new_level > player.level
        if level_up:
            player.level = new_level
        
        # Verificar conquistas
        new_achievements = self._check_achievements(player)
        
        # Atualizar desafios diários
        daily_progress = self._update_daily_challenges(player, is_correct, category)
        
        # Registrar atividade recente
        self._add_recent_activity(player, category, is_correct, points if is_correct else 0, level_up)
        
        return {
            "points": points if is_correct else 0,
            "level_up": level_up,
            "new_level": player.level,
            "new_achievements": new_achievements,
            "daily_progress": daily_progress,
            "streak": player.streak,
            "total_xp": player.experience
        }
    
    def _calculate_level(self, experience: int) -> int:
        """Calcula o nível baseado na experiência"""
        # Fórmula: level = sqrt(experience / 100) + 1
        return int((experience / 100) ** 0.5) + 1
    
    def _check_achievements(self, player: Player) -> List[Achievement]:
        """Verifica conquistas desbloqueadas"""
        new_achievements = []
        
        # Garantir que atributos existam
        if not hasattr(player, 'historic_locations_identified'):
            player.historic_locations_identified = 0
        if not hasattr(player, 'fastest_time') or player.fastest_time == 0:
            player.fastest_time = 999.0
        
        print(f"[ACHIEVEMENT DEBUG] Verificando conquistas para player {player.name}")
        print(f"[ACHIEVEMENT DEBUG] total_attempts={player.total_attempts}, streak={player.streak}, level={player.level}, fastest_time={player.fastest_time}, historic_locations={player.historic_locations_identified}")
        
        for achievement in self.achievements:
            if achievement.id in player.achievements:
                print(f"[ACHIEVEMENT DEBUG] Conquista {achievement.name} já desbloqueada")
                continue
                
            # Verificar condição da conquista
            condition_met = self._evaluate_condition(achievement.condition, player)
            print(f"[ACHIEVEMENT DEBUG] Conquista {achievement.name} (condição: {achievement.condition}) = {condition_met}")
            
            if condition_met:
                player.achievements.append(achievement.id)
                player.experience += achievement.reward_xp
                new_achievements.append(achievement)
                print(f"[ACHIEVEMENT DEBUG] ✓ Nova conquista desbloqueada: {achievement.name} (+{achievement.reward_xp} XP)")
        
        return new_achievements
    
    def _evaluate_condition(self, condition: str, player: Player) -> bool:
        """Avalia uma condição de conquista"""
        # Implementação simples - em produção seria mais robusta
        try:
            # Garantir que todos os atributos necessários existam
            if not hasattr(player, 'historic_locations_identified'):
                player.historic_locations_identified = 0
            if not hasattr(player, 'fastest_time') or player.fastest_time == 0:
                player.fastest_time = 999.0
            
            # Criar contexto com variáveis acessíveis
            # As condições usam variáveis diretas como "total_attempts >= 1"
            # mas precisamos acessar player.total_attempts
            context = {
                'total_attempts': player.total_attempts,
                'streak': player.streak,
                'level': player.level,
                'historic_locations_identified': player.historic_locations_identified,
                'fastest_time': player.fastest_time,
                'total_correct': player.total_correct,
                'experience': player.experience,
            }
            
            # Avaliar condição com contexto seguro
            # Não permitir imports ou funções perigosas
            safe_builtins = {
                'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
                'enumerate': enumerate, 'float': float, 'int': int, 'len': len,
                'list': list, 'max': max, 'min': min, 'range': range,
                'round': round, 'set': set, 'sorted': sorted, 'str': str,
                'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
            }
            
            result = eval(condition, {"__builtins__": safe_builtins}, context)
            return bool(result)
        except Exception as e:
            print(f"[ACHIEVEMENT DEBUG] Erro ao avaliar condição '{condition}': {e}")
            return False
    
    def _add_recent_activity(self, player: Player, category: str, is_correct: bool, points: int, level_up: bool):
        """Adiciona uma atividade recente ao histórico do jogador"""
        import datetime
        
        # Garantir que recent_activities existe
        if not hasattr(player, 'recent_activities') or player.recent_activities is None:
            player.recent_activities = []
        
        # Mapear categorias para ícones e nomes
        category_info = {
            "historic_location": {"icon": "📸", "name": "Modo Foto"},
            "description": {"icon": "✍️", "name": "Modo Descrição"},
            "mystery_mode": {"icon": "🔮", "name": "Modo Mistério"},
            "general": {"icon": "🎮", "name": "Jogo"}
        }
        
        info = category_info.get(category, {"icon": "🎯", "name": "Atividade"})
        
        # Criar atividade
        activity = {
            "timestamp": datetime.datetime.now().isoformat(),
            "category": category,
            "icon": info["icon"],
            "name": info["name"],
            "is_correct": is_correct,
            "points": points,
            "level_up": level_up,
            "streak": player.streak
        }
        
        # Adicionar no início da lista
        player.recent_activities.insert(0, activity)
        
        # Manter apenas as 10 atividades mais recentes
        if len(player.recent_activities) > 10:
            player.recent_activities = player.recent_activities[:10]
    
    def get_recent_activities(self, player_id: str, limit: int = 5) -> List[Dict]:
        """Retorna atividades recentes do jogador"""
        player = self.get_player(player_id)
        if not player:
            return []
        
        if not hasattr(player, 'recent_activities') or not player.recent_activities:
            return []
        
        return player.recent_activities[:limit]
    
    def _update_daily_challenges(self, player: Player, is_correct: bool, 
                                category: str) -> Dict:
        """Atualiza progresso dos desafios diários"""
        progress = {}
        
        for challenge in self.daily_challenges:
            challenge_id = challenge["id"]
            
            if challenge_id not in player.daily_challenges:
                player.daily_challenges[challenge_id] = 0
            
            # Atualizar progresso baseado no tipo de desafio
            if challenge_id == "daily_images":
                player.daily_challenges[challenge_id] += 1
            elif challenge_id == "daily_accuracy" and is_correct:
                player.daily_challenges[challenge_id] += 1
            elif challenge_id == "daily_categories" and category:
                if category not in player.daily_challenges.get("categories_seen", []):
                    if "categories_seen" not in player.daily_challenges:
                        player.daily_challenges["categories_seen"] = []
                    player.daily_challenges["categories_seen"].append(category)
                    player.daily_challenges[challenge_id] += 1
            
            progress[challenge_id] = {
                "current": player.daily_challenges[challenge_id],
                "target": challenge["target"],
                "completed": player.daily_challenges[challenge_id] >= challenge["target"]
            }
        
        return progress
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Retorna o ranking dos jogadores"""
        sorted_players = sorted(
            self.players.values(),
            key=lambda p: (p.level, p.experience),
            reverse=True
        )
        
        leaderboard = []
        for i, player in enumerate(sorted_players[:limit]):
            leaderboard.append({
                "id": player.id,
                "rank": i + 1,
                "name": player.name,
                "level": player.level,
                "experience": player.experience,
                "accuracy": player.total_correct / max(player.total_attempts, 1),
                "streak": player.streak
            })
        
        return leaderboard
    
    def get_player_stats(self, player_id: str) -> Dict:
        """Retorna estatísticas detalhadas do jogador"""
        player = self.get_player(player_id)
        if not player:
            return {"error": "Jogador não encontrado"}
        
        accuracy = player.total_correct / max(player.total_attempts, 1)
        next_level_xp = (player.level ** 2) * 100 - player.experience
        
        # Obter detalhes das conquistas
        achievements_details = []
        for achievement_id in player.achievements:
            for achievement in self.achievements:
                if achievement.id == achievement_id:
                    achievements_details.append({
                        'id': achievement.id,
                        'name': achievement.name,
                        'description': achievement.description,
                        'icon': achievement.icon
                    })
                    break
        
        return {
            "name": player.name,
            "level": player.level,
            "experience": player.experience,
            "next_level_xp": next_level_xp,
            "streak": player.streak,
            "total_correct": player.total_correct,
            "total_attempts": player.total_attempts,
            "accuracy": accuracy,
            "achievements": len(player.achievements),
            "achievements_details": achievements_details,  # Detalhes completos
            "daily_challenges": player.daily_challenges
        }

    def add_experience(self, player_id: str, amount: int) -> Dict:
        """Adiciona experiência arbitrária ao jogador (ex.: bônus de modos especiais).
        Atualiza level e conquistas conforme necessário.
        """
        player = self.get_player(player_id)
        if not player:
            return {"error": "Jogador não encontrado"}
        try:
            add = max(0, int(amount))
        except Exception:
            add = 0
        if add <= 0:
            return {"total_xp": player.experience, "level_up": False, "new_level": player.level, "new_achievements": []}

        player.experience += add
        new_level = self._calculate_level(player.experience)
        level_up = new_level > player.level
        if level_up:
            player.level = new_level
        new_achievements = self._check_achievements(player)
        return {
            "total_xp": player.experience,
            "level_up": level_up,
            "new_level": player.level,
            "new_achievements": new_achievements
        }
    
    def reset_daily_challenges(self):
        """Reseta os desafios diários (chamado diariamente)"""
        self.daily_challenges = self._generate_daily_challenges()
        for player in self.players.values():
            player.daily_challenges = {}

# Exemplo de uso
if __name__ == "__main__":
    # Criar sistema de gamificação
    game_system = GamificationSystem()
    
    # Criar jogador
    player = game_system.create_player("player1", "João")
    
    # Simular algumas respostas
    results = []
    results.append(game_system.submit_answer("player1", True, 15.5, "historic_location"))
    results.append(game_system.submit_answer("player1", True, 12.3, "historic_location"))
    results.append(game_system.submit_answer("player1", False, 20.1, "historic_location"))
    
    # Ver estatísticas
    stats = game_system.get_player_stats("player1")
    print("Estatísticas do jogador:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Ver ranking
    leaderboard = game_system.get_leaderboard()
    print("\nRanking:")
    print(json.dumps(leaderboard, indent=2, ensure_ascii=False))

