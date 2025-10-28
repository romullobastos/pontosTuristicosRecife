import json
import os
import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import random

# NLP Libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("NLP libraries not available. Using basic comparison.")

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
        
        # Inicializar NLP
        if NLP_AVAILABLE:
            self._initialize_nlp()
            self.use_nlp = True
        else:
            self.use_nlp = False
    
    def _initialize_nlp(self):
        """Inicializa recursos do NLTK"""
        try:
            print("\n[INFO] Inicializando NLP...")
            # Download recursos necessários
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            self.stop_words = set(stopwords.words('portuguese'))
            self.lemmatizer = WordNetLemmatizer()
            self.vectorizer = TfidfVectorizer(max_features=1000)
            
            print("[OK] NLP inicializado com sucesso!")
            print(f"   - Stopwords carregadas: {len(self.stop_words)} palavras")
            print(f"   - Método de similaridade: TF-IDF + Cosseno")
        except Exception as e:
            print(f"[ERRO] Erro ao inicializar NLP: {e}")
            self.use_nlp = False
    
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
    
    def set_current_photo(self, photo_id: str) -> bool:
        """Define a foto atual pelo ID"""
        if not self.photos_data:
            return False
        
        for photo in self.photos_data:
            if photo["id"] == photo_id:
                self.current_photo = photo
                return True
        
        return False
    
    def submit_description(self, user_description: str) -> Dict:
        """
        Compara a descrição do usuário com a descrição oficial
        """
        if not self.current_photo:
            return {"error": "Nenhuma foto selecionada"}
        
        # Limpar e normalizar descrições
        user_desc_clean = self._clean_text(user_description)
        official_desc_clean = self._clean_text(self.current_photo["official_description"])
        
        # Processar com NLP se disponível
        if self.use_nlp:
            user_desc_processed = self._preprocess_nlp(user_desc_clean)
            official_desc_processed = self._preprocess_nlp(official_desc_clean)
            
            # Debug: mostrar processamento NLP
            print(f"\n[NLP DEBUG]")
            print(f"   Original: {user_desc_clean[:100]}...")
            print(f"   Processado: {user_desc_processed[:100]}...")
        else:
            user_desc_processed = user_desc_clean
            official_desc_processed = official_desc_clean
        
        # Calcular similaridade
        similarity_score = self._calculate_similarity(user_desc_processed, official_desc_processed)
        
        # Verificar palavras-chave
        keyword_score = self._check_keywords(user_desc_clean, self.current_photo["keywords"])
        
        # Calcular pontuação final
        final_score = self._calculate_final_score(similarity_score, keyword_score)
        
        # Determinar se acertou (threshold de 40% - mais acessível)
        is_correct = final_score >= 0.4
        
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
    
    def _preprocess_nlp(self, text: str) -> str:
        """Processa texto com NLP (tokenização, lemmatização, remoção de stopwords)"""
        if not self.use_nlp:
            return text
        
        try:
            # Tokenizar usando split simples (mais robusto que NLTK para português)
            tokens = re.findall(r'\b\w+\b', text)
            
            # Remover stopwords e lemmatizar
            processed_tokens = []
            for token in tokens:
                token_lower = token.lower()
                if token_lower not in self.stop_words and len(token_lower) > 2:
                    lemma = self.lemmatizer.lemmatize(token_lower)
                    if len(lemma) > 2:  # Filtrar palavras muito curtas
                        processed_tokens.append(lemma)
            
            return ' '.join(processed_tokens)
        except Exception as e:
            print(f"Erro no preprocessing NLP: {e}")
            return text
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade entre dois textos"""
        if self.use_nlp:
            # Usar TF-IDF + Cosseno para similaridade semântica
            try:
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            except Exception as e:
                print(f"Erro no cálculo TF-IDF: {e}")
                # Fallback para SequenceMatcher
                return SequenceMatcher(None, text1, text2).ratio()
        else:
            # Usar SequenceMatcher (básico)
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
        final_score = (similarity * 0.6) + (keyword_score * 0.4)
        
        # Debug: mostrar detalhes do cálculo
        print(f"\n[SCORE DEBUG]")
        print(f"   Similaridade: {similarity:.2%}")
        print(f"   Keywords: {keyword_score:.2%}")
        print(f"   Score final: {final_score:.2%}")
        print(f"   Threshold: 40% (para ganhar pontos)")
        
        return final_score
    
    def _calculate_points(self, final_score: float, is_correct: bool) -> int:
        """Calcula pontos ganhos baseado na pontuação"""
        base_points = self.current_photo["points"]
        
        # Dar pontos proporcionais mesmo se não acertou completamente
        if is_correct:
            # Se passou do threshold, dar pontos completos
            if final_score >= 0.9:
                multiplier = 1.5  # Excelente
            elif final_score >= 0.8:
                multiplier = 1.3  # Muito bom
            elif final_score >= 0.7:
                multiplier = 1.1  # Bom
            elif final_score >= 0.6:
                multiplier = 1.0  # Básico
            else:
                multiplier = 0.8  # Ainda ganha, mas menos pontos
        else:
            # Dar pontos parciais proporcionalmente
            # Ex: 30% de score = 30% dos pontos base
            multiplier = max(0.1, final_score)  # Mínimo 10% dos pontos
        
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
            return "Boa tentativa! Continue melhorando seus detalhes. Descreva mais características da foto!"
        elif final_score >= 0.3:
            return "Você está no caminho certo! Tente descrever elementos arquitetônicos, cores e localização."
        elif final_score >= 0.2:
            return "Continue tentando! Preste atenção nos detalhes da arquitetura e ambiente."
        else:
            return "Não desista! Observe melhor a foto e tente mencionar o tipo de local, época e características visuais."
    
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
