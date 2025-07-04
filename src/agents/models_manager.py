# models_manager.py
import os
import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from typing import List, Dict, Union

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek


load_dotenv()

class ModelConfig:
    def __init__(self, display_name: str, model_name: str, provider: str):
        self.display_name = display_name
        self.model_name = model_name
        self.provider = provider

def get_chat_model(config: ModelConfig) -> Union[ChatOpenAI, ChatGroq, ChatDeepSeek]:
    """Создает экземпляр чат-модели по конфигурации"""
    if config.provider.lower() == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found in .env")
        return ChatGroq(model_name=config.model_name, api_key=api_key)
    elif config.provider.lower() == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in .env")
        return ChatOpenAI(model=config.model_name, api_key=api_key)
    elif config.provider.lower() == 'deepseek':
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found in .env")
        return ChatDeepSeek(model=config.model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

def load_model_configs(config_path: Path) -> List[ModelConfig]:
    """Загружает конфигурации моделей из JSON файла"""
    try:
        with open(config_path, 'r') as f:
            models_data = json.load(f)
        return [ModelConfig(**data) for data in models_data]
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file: {config_path}")
        raise

def initialize_chat_models(configs: List[ModelConfig]) -> Dict[str, Union[ChatOpenAI, ChatGroq, ChatDeepSeek]]:
    """Инициализирует все чат-модели по конфигурациям"""
    models = {}
    for config in configs:
        try:
            models[config.display_name] = get_chat_model(config)
            logger.info(f"Model initialized: {config.display_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {config.display_name}: {e}")
    return models