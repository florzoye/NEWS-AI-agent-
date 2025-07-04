# news_classifier.py
from typing import List, Dict, Optional, Union
from loguru import logger

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough




class NewsClassifier:
    def __init__(self, chat_model: Union[ChatOpenAI, ChatGroq, ChatDeepSeek], model_name: str):
        self.chat_model = chat_model
        self.model_name = model_name

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """
            Ты — ведущий финансовый аналитик хедж-фонда с 15-летним опытом. 
            Анализируй новости как для краткосрочного трейдинга, так и для долгосрочных инвестиций
            

            ### Задачи:
            Классифицируй новости на 3 группы:
            1. Классификация:
            - Криптовалюты (crypto): Биткоин, альткоины, блокчейн, регулирование, ETF и все в этом духе - 0 группа
            - Экономика (economy): Макроэкономика, ЦБ, инфляция, рынки акций/облигаций, крах/рост компании, все в этом духе - 1 группа
            - Другое (other): Всё остальное - 2 группа

            2. Анализ:
            - Выдели несколько ключевых тренда из новостей
            - Оцени влияние на:
            * Крипторынок (1-7 дней)
            * Фондовый рынок (1-3 месяца)
            * Макроэкономику (6-12 месяцев)

            3. Торговые идеи:
            - Для крипторынка: возможные фьючерс/спот сделки, уход/приход волатильности на монете, мелкие спекуляции
            - Для акций: возможность покупки/продажи акций определенных компаний, так же бери в учет стратегию Бена Грахама, Фила Фишера и Уоренна Баффета
            - Риск-менеджмент: оцени риск сделки используя логические и реальные аргументы, в процентах

            ### Формат ответа:
            {{
                "analysis": "Текст с глубоким анализом (макс. 500 символов), исходя из задач, оформляй в произвольном стиле",
                "crypto_trading_ideas": [
                    {{
                        "asset": "сюда указывай монету (например SOl/USD, TRUMP/SOL и тд)",
                        "direction": "здесь указывай позицию (long, short)",
                        "timeframe": "здесь интервал сколько держать сделку",
                        "reason": "здесь рассуждения"
                    }}
                ],
                "stock_traiding_ideas": [
                    {{
                        "company_ticker": "укажи тикер компании",
                        "signal": "сигнал на покупку/продажу",
                        "idea": "основная идея твоего сигнала",

                    }}
                ],
                "classification": {{
                    "crypto": [номера новости исходя из предоставленного списка],
                    "economy": [номера новости исходя из предоставленного списка],
                    "other": [номера новости исходя из предоставленного списка]
                }}
            }}

            ### Новости:
            {news}
            """
            )

    async def classify_news(self, news: List[str]) -> Optional[Dict]:
        try:
            numbered_news = [f"{i}: {text}" for i, text in enumerate(news)]
            
            chain = (
                {"news": RunnablePassthrough()} 
                | self.prompt_template
                | self.chat_model
                | JsonOutputParser()
            )
            
            result = await chain.ainvoke("\n".join(numbered_news))
            return {
                "model": self.model_name,
                "result": result
            }
        except Exception as e:
            logger.error(f"Classification error in {self.model_name}: {e}")
            return None