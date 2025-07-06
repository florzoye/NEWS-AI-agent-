import asyncio
from telethon.sync import TelegramClient, events
from PIL import Image
import io
import logging
from typing import Optional
import re
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from db.newsDB import NewsSQL
from db.manager import DatabaseManager
from src.utils.config import api_hash, api_id, phone_number
from src.agents.crypto_filter import CryptoClassifierTester

BASE_DIR = Path(__file__).parent.parent  # Для стандартной структуры проекта



# Пути к файлам относительно BASE_DIR
PATHS = {
    "embeddings": BASE_DIR / "data" / "other" / "navec_crypto_extended.npz",
    "crypto_news": BASE_DIR / "data" / "news_cat" / "crypto_news.txt",
    "economy_news": BASE_DIR / "data" / "news_cat" / "economy_news.txt",
    "other_news": BASE_DIR / "data" / "news_cat" / "other_news.txt",
    "model_save": BASE_DIR / "data" / "other" / "crypto_classifier.pt"
}

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramNewsBot:
    def __init__(self):
        self.client = TelegramClient('news_session', api_id, api_hash)
        self.client.add_event_handler(
            self.handle_message,
            events.NewMessage(incoming=True)
        )
        self.db_manager = DatabaseManager(db_path='database.db')
        self._init_db()

        self.chanels_names = []

    def _init_db(self):
        """Инициализация таблицы в базе данных"""
        with self.db_manager.get_cursor() as cursor:
            NewsSQL(cursor).create_table('NEWS_text')

    @staticmethod
    async def _clean_data(text: str) -> str:
        # Удаляем все символы, кроме букв, цифр и пробелов
        cleaned_text = re.sub(r'^[a-zA-Z0-9\s]+$', '', text, flags=re.UNICODE)
        
        # Удаляем оставшиеся символы подчеркивания (если нужно)
        cleaned_text = cleaned_text.replace('_', ' ')
        
        # Удаляем лишние пробелы и приводим к нижнему регистру
        cleaned_text = ' '.join(cleaned_text.lower().split())
        return cleaned_text
    
    async def classification_news(self, news: str):
        tester = CryptoClassifierTester(
            model_path=PATHS['model_save'],
            embeddings_path=PATHS["embeddings"],
        )
        result = tester.predict(news)
        if result['class'] == "Криптовалюта":
            return 'crypto'
        elif result['class'] == 'Экономика':
            return 'economy'
        else:
            return 'other'
    
    async def process_content(self, text: str, source: str) -> Optional[dict]:
        """Обрабатывает текст через весь пайплайн."""
        try:
            clean_text = await self._clean_data(text)
            if not clean_text:
                return None
            
            news_class = await self.classification_news(news=clean_text)
            news_item = {
                'source': source,
                'text': clean_text,  
                'class_news': news_class 
            }
            
            with self.db_manager.get_cursor() as cursor:
                NewsSQL(cursor).add_info('NEWS_text', news_item)
            
            return {'clean_text': clean_text}
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None
    
    async def handle_message(self, event):
        """Основной обработчик сообщений."""
        try:
            sender = await event.get_sender()
            logger.info(f"New message from {sender.id} ({getattr(sender, 'username', 'No username')})")

            name = getattr(sender, 'username', 'No username')
            if name:
                if name not in self.chanels_names:
                    self.chanels_names.append(name)
            if event.text:
                await self.process_content(event.text, 'telegram_text')
        
        except Exception as e:
            logger.error(f"Handler error: {e}")

    async def clean_db(self):
        with self.db_manager.get_cursor() as cursor:
            NewsSQL(cursor).clear_table('NEWS_text')
            

    async def run(self):
        """Запускает бота."""
        await self.client.start(phone_number)
        logger.info("Bot started. Waiting for messages...")
        await self.client.run_until_disconnected()

if __name__ == "__main__":
    try:
        bot = TelegramNewsBot()
        # asyncio.run(bot.clean_db())
        asyncio.run(bot.run())
    except EnvironmentError:
        logger.error("Tesseract OCR not installed! Please install it first.")
        exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        exit(0)