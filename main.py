# main.py
import json
from pathlib import Path
from typing import Dict, List, Union

import sqlite3
import pandas as pd

import asyncio
from loguru import logger

from  src.agents.money_maker_agent import NewsClassifier
from src.agents.models_manager import load_model_configs, initialize_chat_models

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek


def load_news_from_db(db_path: Path, limit: int = None) -> List[str]:
    """Загружает новости из базы данных"""
    try:
        with sqlite3.connect(db_path) as con:
            query = "SELECT text FROM NEWS_text"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql(query, con=con)
            return df['text'].tolist()
    except Exception as e:
        logger.error(f"Ошибка загрузки новостей: {e}")
        raise

async def process_news_with_all_models(
    news: List[str],
    chat_models: Dict[str, Union[ChatOpenAI, ChatGroq, ChatDeepSeek]]
) -> List[Dict]:
    """Обрабатывает новости всеми моделями параллельно"""
    classifiers = [
        NewsClassifier(model, name) 
        for name, model in chat_models.items()
    ]
    
    tasks = [
        classifier.classify_news(news)
        for classifier in classifiers
    ]
    
    return await asyncio.gather(*tasks)

async def main():
    try:
        # Конфигурация путей
        config_path = Path(__file__).parent / "src" / "agents" / "api_models.json"
        
        # Загрузка конфигураций моделей
        model_configs = load_model_configs(config_path)
        chat_models = initialize_chat_models(model_configs)

        news = load_news_from_db('database.db', limit=2)  # Пример с лимитом
        
        # # Обработка новостей
        results = await process_news_with_all_models(news, chat_models)
        
        # Вывод результатов
        for result in results:
            if result:
                print(f"\nResults from {result['model']}:")
                print(json.dumps(result['result'], indent=2, ensure_ascii=False))
                
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    asyncio.run(main())