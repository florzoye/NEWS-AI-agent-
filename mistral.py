import os
import sqlite3
from pathlib import Path
from typing import List
import pandas as pd
from mistralai import Mistral
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def analyze_news_with_mistral(news_items: List[str]) -> dict:
    """Анализирует новости с помощью Mistral AI"""
    try:
        api_key = os.environ["MISTRAL_API_KEY"]
        model = "mistral-large-latest"
        
        client = Mistral(api_key=api_key)
        
        # Объединяем новости в один текст с нумерацией
        news_text = "\n".join([f"{i+1}. {news}" for i, news in enumerate(news_items)])
        
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Ты — ведущий финансовый аналитик хедж-фонда с 15-летним опытом. 
                    Анализируй новости как для краткосрочного трейдинга, так и для долгосрочных инвестиций

                    ### Задачи:
                    1. Анализ:
                    - Выдели несколько ключевых тренда из новостей
                    - Оцени влияние на:
                    * Крипторынок (1-7 дней)
                    * Фондовый рынок (1-3 месяца)
                    * Макроэкономику (6-12 месяцев)

                    2. Торговые идеи:
                    - Для крипторынка: возможные фьючерс/спот сделки, уход/приход волатильности на монете, мелкие спекуляции
                    - Для акций: возможность покупки/продажи акций определенных компаний, учитывая стратегии Бена Грахама, Фила Фишера и Уоррена Баффета
                    - Риск-менеджмент: оцени риск сделки используя логические и реальные аргументы

                    ### Формат ответа (строго в JSON):
                    {{
                        "analysis": "Текст с глубоким анализом (не ограничивайся в количестве символов, сделай разбор с дополнительными и основными активностями)",
                        "crypto_trading_ideas": [
                            {{
                                "asset": "пара криптовалют (например BTC/USD)",
                                "direction": "long/short/buy/sell",
                                "timeframe": "временной горизонт",
                                "reason": "обоснование",
                                "risk_percentage": "оценка риска 1-100%"
                            }}
                        ],
                        "stock_trading_ideas": [
                            {{
                                "company_ticker": "тикер компании",
                                "signal": "buy/sell/hold",
                                "idea": "обоснование",
                                "risk_percentage": "оценка риска 1-100%"
                            }}
                        ],
                    }}

                    ### Новости:
                    {news_text}
                    """
                }
            ]
        )
        
        return chat_response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Ошибка анализа новостей: {e}")
        raise

def main():
    # Путь к базе данных
    db_path = Path("database.db")
    
    try:
        # Загрузка новостей (первые 10)
        news_items = load_news_from_db(db_path)
        logger.info(f"Загружено {len(news_items)} новостей")
        
        # Анализ новостей
        analysis_result = analyze_news_with_mistral(news_items)
        logger.info("Анализ завершен успешно")
        
        # Вывод результата
        print("Результат анализа:")
        print(analysis_result)
        
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")

if __name__ == "__main__":
    main()