 
# Store recognized signs
sentence_words = []

def add_sign(sign):
    if sign != "UNKNOWN" and sign != "SHOW YOUR HAND":
        sentence_words.append(sign)

def get_sentence():
    return " ".join(sentence_words)

def clear_sentence():
    sentence_words.clear()
