import numpy as np
import re
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import logging

import os
from pathlib import Path

# Путь к директории проекта
BASE_DIR = Path(__file__).parent.parent  # Для стандартной структуры проекта



# Пути к файлам относительно BASE_DIR
PATHS = {
    "embeddings": BASE_DIR / "data" / "other" / "navec_crypto_extended.npz",
    "crypto_news": BASE_DIR / "data" / "news_cat" / "crypto_news.txt",
    "economy_news": BASE_DIR / "data" / "news_cat" / "economy_news.txt",
    "other_news": BASE_DIR / "data" / "news_cat" / "other_news.txt",
    "model_save": BASE_DIR / "data" / "other" / "crypto_classifier.pt"
}

# Проверка существования файлов
for name, path in PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"Файл {name} не найден по пути: {path}")

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEmbeddings:
    def __init__(self, npz_path):
        logger.info(f"Loading embeddings from {npz_path}")
        try:
            data = np.load(npz_path)
            self.embeddings = {word: vec for word, vec in zip(data['vocab_words'], data['vectors'])}
            logger.info(f"Loaded {len(self.embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def __getitem__(self, word):
        return self.embeddings.get(word, np.zeros(300))
    
    def __contains__(self, word):
        return word in self.embeddings



class PhraseDataset(torch.utils.data.Dataset):
    def __init__(self, paths_dict, embeddings, min_length=1):
        logger.info("Initializing PhraseDataset")
        self.embeddings = embeddings
        self.min_length = min_length
        self.phrases = []
        self.labels = []
        
        # Загрузка и обработка данных
        for label, (category, path) in enumerate(paths_dict.items()):
            if not Path(path).exists():
                raise FileNotFoundError(f"Файл для категории {category} не найден: {path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    phrase = self._clean_phrase(line)
                    if len(phrase) >= min_length:
                        self.phrases.append(phrase)
                        self.labels.append(label)
        
        if not self.phrases:
            logger.error("No valid phrases found after filtering!")
            raise ValueError("No valid phrases found after filtering!")
        else:
            logger.info(f"Total phrases loaded: {len(self.phrases)}")

    def _clean_phrase(self, text):
        text = text.lower().strip()
        words = [w for w in text.split() if w in self.embeddings]
        return words

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        words = self.phrases[idx]
        embeddings = [torch.tensor(self.embeddings[w], dtype=torch.float32) for w in words]
        return torch.stack(embeddings), self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        batch = [x for x in batch if len(x[0]) > 0]  # Фильтрация пустых последовательностей
        if not batch:
            return torch.zeros(0, 0, 300), torch.zeros(0), torch.zeros(0)
        
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in sequences])
        padded_seqs = pad_sequence(sequences, batch_first=True)
        return padded_seqs, torch.tensor(labels), lengths

class CryptoClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        logger.info(f"Initializing model with input_size={input_size}, hidden_size={hidden_size}")
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, lengths):
        if x.size(0) == 0:  # Пустой батч
            return torch.zeros(0, self.fc.out_features, device=x.device)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        outputs, hidden = self.rnn(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)

def train():
    logger.info("Starting training process")
    
    try:
        # 1. Загрузка эмбеддингов
        embeddings = CustomEmbeddings(PATHS["embeddings"])
        
        paths_dict = {
            'crypto': PATHS["crypto_news"],
            'economy': PATHS["economy_news"],
            'other': PATHS["other_news"]
        }
        
        dataset = PhraseDataset(paths_dict, embeddings, min_length=3)
        
        # 3. Создание DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=PhraseDataset.collate_fn
        )
        
        # 4. Инициализация модели
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = CryptoClassifier(
            input_size=300,
            hidden_size=128,
            num_classes=3
        ).to(device)
        
        # 5. Обучение
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        logger.info("Starting training loop")
        for epoch in range(35):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for seqs, labels, lengths in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                if seqs.size(0) == 0:  # Пропускаем пустые батчи
                    continue
                    
                seqs, labels = seqs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(seqs, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                logger.info(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
            else:
                logger.warning(f'Epoch {epoch+1} had no valid batches!')
        
        torch.save(model.state_dict(), PATHS["model_save"])
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)



class CryptoClassifierTester:
    def __init__(self, model_path, embeddings_path):
        # Загрузка модели
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CryptoClassifier(
            input_size=300,
            hidden_size=128,
            num_classes=3
        ).to(self.device)

        if not PATHS["model_save"].exists():
            raise FileNotFoundError(f"Модель не найдена по пути: {PATHS['model_save']}")
        
        # Загрузка весов модели
        self.model.load_state_dict(torch.load(PATHS["model_save"], map_location=self.device))
        self.model.eval()
        
        # Загрузка эмбеддингов
        self.embeddings = CustomEmbeddings(PATHS["embeddings"])
        
        # Метки классов
        self.class_names = {0: "Криптовалюта", 1: "Экономика", 2: "Другое"}

    def predict(self, text):
        # Преобразование текста в эмбеддинги
        words = [w for w in text.lower().split() if w in self.embeddings]
        if not words:
            return "Нет известных слов в тексте"
        
        embeddings = [torch.tensor(self.embeddings[w], dtype=torch.float32) for w in words]
        seq = torch.stack(embeddings).unsqueeze(0).to(self.device)  # Добавляем размерность батча
        lengths = torch.tensor([len(words)], device=self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(seq, lengths)
            probs = torch.softmax(outputs, dim=1)
            class_idx = torch.argmax(probs).item()
            confidence = probs[0, class_idx].item()
        
        return {
            "class": self.class_names[class_idx],
            "confidence": round(confidence, 3),
            "probabilities": {
                self.class_names[i]: round(prob.item(), 3) 
                for i, prob in enumerate(probs[0])
            }
        }
    


if __name__ == '__main__':
    # Инициализация тестера
    # train()
    tester = CryptoClassifierTester(
        model_path='crypto_classifier.pt',
        embeddings_path='navec_crypto_extended.npz'
    )
    # 
    # Тестовые фразы
    test_phrases = [
        'Ukraine failed to meet EU accession criteria — Polish president',
        'Limitless, рынок прогнозов на ETH, привлек 4 млн $, а Артур Хейс присоединился к компании в качестве консультанта наряду с инвестициями из его семейного офиса Maelstrom. ',
        'долларовый стейблкоин **global dollar (usdg)** запущен в ес с полным соответствием mica. стейблкоин поддерживается компаниями kraken, robinhood и mastercard и теперь доступен более чем 450 миллионам потребителей в 30 европейских странах'
    ]
    # 
    # Проверка предсказаний
    print("\nРезультаты тестирования модели:")
    for phrase in test_phrases:
        result = tester.predict(phrase)
        print(f"\nФраза: '{phrase}'")
        print(f"Класс: {result['class']} (уверенность: {result['confidence']*100:.1f}%)")
        print("Распределение вероятностей:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob*100:.1f}%")