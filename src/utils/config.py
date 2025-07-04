from dotenv import load_dotenv
import os

load_dotenv()  

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
phone_number = os.getenv("PHONE_NUMBER")
device = os.getenv('device')
openai_key = os.getenv('OPENAI_KEY')