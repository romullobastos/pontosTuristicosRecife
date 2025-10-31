#!/usr/bin/env python3
"""
Reseta o ranking/jogadores do jogo, limpando o arquivo data/players.json.
Uso (Windows PowerShell):
  python scripts/reset_players.py
"""
import os
import sys

# Permitir importar o pacote a partir da raiz do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from game.gamification import GamificationSystem


def main():
    gs = GamificationSystem()
    gs.reset_players('data/players.json')
    print('✅ Jogadores e ranking foram resetados (data/players.json).')


if __name__ == '__main__':
    main()
