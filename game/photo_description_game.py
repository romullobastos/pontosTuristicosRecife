import json
import os
import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import random

class PhotoDescriptionGame:
    """
    Sistema de jogo baseado em descrições de fotos históricas do Recife
    """
    
    def __init__(self):
        self.photos_data = self._load_photos_data()
        self.current_photo = None
        self.game_stats = {
            "total_attempts": 0,
            "total_correct": 0,
            "total_points": 0,
            "current_streak": 0,
            "best_streak": 0
        }
    
    def _load_photos_data(self) -> List[Dict]:
        """Carrega os dados das fotos históricas"""
        try:
            with open('data/photo_descriptions.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Arquivo photo_descriptions.json não encontrado!")
            return []
    
    def get_random_photo(self) -> Dict:
        """Retorna uma foto aleatória para o jogo"""
        if not self.photos_data:
            return None
        
        self.current_photo = random.choice(self.photos_data)
        return {
            "id": self.current_photo["id"],
            "image_path": self.current_photo["image_path"],
            "name": self.current_photo["name"],
            "difficulty": self.current_photo["difficulty"],
            "points": self.current_photo["points"]
        }
    
    def submit_description(self, user_description: str) -> Dict:
        """
        Compara a descrição do usuário com a descrição oficial
        """
        if not self.current_photo:
            return {"error": "Nenhuma foto selecionada"}
        
        # Limpar e normalizar descrições
        user_desc_clean = self._clean_text(user_description)
        official_desc_clean = self._clean_text(self.current_photo["official_description"])
        
        # Calcular similaridade
        similarity_score = self._calculate_similarity(user_desc_clean, official_desc_clean)
        
        # Verificar palavras-chave
        keyword_score = self._check_keywords(user_desc_clean, self.current_photo["keywords"])
        
        # Calcular pontuação final
        final_score = self._calculate_final_score(similarity_score, keyword_score)
        
        # Determinar se acertou (threshold de 60%)
        is_correct = final_score >= 0.6
        
        # Atualizar estatísticas
        self._update_stats(is_correct)
        
        # Calcular pontos ganhos
        points_earned = self._calculate_points(final_score, is_correct)
        
        return {
            "success": True,
            "is_correct": is_correct,
            "similarity_score": similarity_score,
            "keyword_score": keyword_score,
            "final_score": final_score,
            "points_earned": points_earned,
            "official_description": self.current_photo["official_description"],
            "keywords_found": self._get_found_keywords(user_desc_clean, self.current_photo["keywords"]),
            "feedback": self._generate_feedback(final_score, is_correct),
            "game_stats": self.game_stats.copy()
        }
    
    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza texto para comparação"""
        # Converter para minúsculas
        text = text.lower()
        
        # Remover acentos e caracteres especiais
        text = re.sub(r'[àáâãäå]', 'a', text)
        text = re.sub(r'[èéêë]', 'e', text)
        text = re.sub(r'[ìíîï]', 'i', text)
        text = re.sub(r'[òóôõö]', 'o', text)
        text = re.sub(r'[ùúûü]', 'u', text)
        text = re.sub(r'[ç]', 'c', text)
        
        # Remover pontuação e caracteres especiais
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade entre dois textos usando SequenceMatcher"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _check_keywords(self, user_text: str, keywords: List[str]) -> float:
        """Verifica quantas palavras-chave foram encontradas"""
        if not keywords:
            return 0.0
        
        found_keywords = 0
        for keyword in keywords:
            if keyword.lower() in user_text:
                found_keywords += 1
        
        return found_keywords / len(keywords)
    
    def _calculate_final_score(self, similarity: float, keyword_score: float) -> float:
        """Calcula pontuação final combinando similaridade e palavras-chave"""
        # Peso: 60% similaridade + 40% palavras-chave
        return (similarity * 0.6) + (keyword_score * 0.4)
    
    def _calculate_points(self, final_score: float, is_correct: bool) -> int:
        """Calcula pontos ganhos baseado na pontuação"""
        if not is_correct:
            return 0
        
        base_points = self.current_photo["points"]
        
        # Multiplicador baseado na qualidade da resposta
        if final_score >= 0.9:
            multiplier = 1.5  # Excelente
        elif final_score >= 0.8:
            multiplier = 1.3  # Muito bom
        elif final_score >= 0.7:
            multiplier = 1.1  # Bom
        else:
            multiplier = 1.0  # Básico
        
        return int(base_points * multiplier)
    
    def _get_found_keywords(self, user_text: str, keywords: List[str]) -> List[str]:
        """Retorna palavras-chave encontradas na descrição do usuário"""
        found = []
        for keyword in keywords:
            if keyword.lower() in user_text:
                found.append(keyword)
        return found
    
    def _generate_feedback(self, final_score: float, is_correct: bool) -> str:
        """Gera feedback baseado na pontuação"""
        if final_score >= 0.9:
            return "Excelente! Sua descrição capturou muito bem os elementos principais da foto!"
        elif final_score >= 0.8:
            return "Muito bom! Você identificou corretamente a maioria dos elementos importantes."
        elif final_score >= 0.7:
            return "Bom trabalho! Você acertou vários aspectos da foto histórica."
        elif final_score >= 0.6:
            return "Parabéns! Você conseguiu identificar alguns elementos corretos."
        elif final_score >= 0.4:
            return "Continue tentando! Você está no caminho certo, mas pode melhorar."
        else:
            return "Não desista! Observe melhor os detalhes da foto e tente novamente."
    
    def _update_stats(self, is_correct: bool):
        """Atualiza estatísticas do jogo"""
        self.game_stats["total_attempts"] += 1
        
        if is_correct:
            self.game_stats["total_correct"] += 1
            self.game_stats["current_streak"] += 1
            self.game_stats["best_streak"] = max(
                self.game_stats["best_streak"], 
                self.game_stats["current_streak"]
            )
        else:
            self.game_stats["current_streak"] = 0
    
    def get_game_stats(self) -> Dict:
        """Retorna estatísticas do jogo"""
        accuracy = 0
        if self.game_stats["total_attempts"] > 0:
            accuracy = self.game_stats["total_correct"] / self.game_stats["total_attempts"]
        
        return {
            **self.game_stats,
            "accuracy": accuracy
        }
    
    def reset_game(self):
        """Reseta o jogo"""
        self.current_photo = None
        self.game_stats = {
            "total_attempts": 0,
            "total_correct": 0,
            "total_points": 0,
            "current_streak": 0,
            "best_streak": 0
        }
    
    def get_photo_by_id(self, photo_id: str) -> Dict:
        """Retorna uma foto específica pelo ID"""
        for photo in self.photos_data:
            if photo["id"] == photo_id:
                return photo
        return None

# Exemplo de uso
if __name__ == "__main__":
    game = PhotoDescriptionGame()
    
    # Obter foto aleatória
    photo = game.get_random_photo()
    print(f"Foto: {photo['name']}")
    print(f"Dificuldade: {photo['difficulty']}")
    
    # Simular descrição do usuário
    user_desc = "Esta é uma praça histórica com calçada de pedra portuguesa e vista para o mar"
    result = game.submit_description(user_desc)
    
    print(f"\nResultado:")
    print(f"Acertou: {result['is_correct']}")
    print(f"Pontuação: {result['final_score']:.2f}")
    print(f"Pontos ganhos: {result['points_earned']}")
    print(f"Feedback: {result['feedback']}")
