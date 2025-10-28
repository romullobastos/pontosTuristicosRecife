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
    
    def __post_init__(self):
        if self.achievements is None:
            self.achievements = []
        if self.daily_challenges is None:
            self.daily_challenges = {}

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
    
    def get_player(self, player_id: str) -> Optional[Player]:
        """Retorna um jogador pelo ID"""
        return self.players.get(player_id)
    
    def submit_answer(self, player_id: str, is_correct: bool, 
                     response_time: float, category: str = None) -> Dict:
        """Processa uma resposta do jogador"""
        player = self.get_player(player_id)
        if not player:
            return {"error": "Jogador não encontrado"}
        
        # Atualizar estatísticas
        player.total_attempts += 1
        if is_correct:
            player.total_correct += 1
            player.streak += 1
            
            # Calcular pontos (agora só XP, sem coins)
            base_points = 10
            speed_bonus = max(0, 20 - response_time)  # Bônus por velocidade
            streak_bonus = min(player.streak * 2, 20)  # Bônus por sequência
            
            points = base_points + speed_bonus + streak_bonus
            player.experience += points
            
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
        
        for achievement in self.achievements:
            if achievement.id in player.achievements:
                continue
                
            # Verificar condição da conquista
            if self._evaluate_condition(achievement.condition, player):
                player.achievements.append(achievement.id)
                player.experience += achievement.reward_xp
                new_achievements.append(achievement)
        
        return new_achievements
    
    def _evaluate_condition(self, condition: str, player: Player) -> bool:
        """Avalia uma condição de conquista"""
        # Implementação simples - em produção seria mais robusta
        try:
            return eval(condition, {"player": player})
        except:
            return False
    
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
            "daily_challenges": player.daily_challenges
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

