import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)              
    tokens = [w for w in text.split() if w not in stopwords]
    return " ".join(tokens)
