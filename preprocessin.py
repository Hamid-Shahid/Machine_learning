import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

def preprocess_text(input_text):
    # Tokenization
    words = word_tokenize(input_text.lower())  # Convert text to lowercase and tokenize
    
    words = [word for word in words if word not in string.punctuation]

    # Stop-word removal
    words = [word for word in words if word not in stop_words]  # Remove stop words
    
    # Stemming
    stemmed_words = [porter_stemmer.stem(word) for word in words]  # Stemming
    if len(stemmed_words) >= 300:
        stemmed_words=stemmed_words[:300]
    return ' '.join(stemmed_words)  # Join the preprocessed words back into a string

def process_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        input_text = file.read()
    
    # Preprocess the text
    preprocessed_text = preprocess_text(input_text)
    
    output_file = os.path.splitext(input_file)[0] + '_preprocessed.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(preprocessed_text)
    
    print("Text preprocessed and written to '{}'.".format(output_file))

input_folder = 'Business and Finance data'
input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.txt')]
for input_file in input_files:
    process_file(input_file)
