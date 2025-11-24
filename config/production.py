# Configuração para Produção

import os

class ProductionConfig:
    """Configuração para ambiente de produção"""
    
    # Flask
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'change-this-in-production')
    
    # Server
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Database
    DATABASE_PATH = 'data/app.db'
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS
    CORS_ORIGINS = ['http://seu-dominio.com', 'https://seu-dominio.com']
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/production.log'
