import os

def remove_duplicates(input_text):
    unique_words = set()
    words = input_text.split()
    unique_words.update(word.lower() for word in words)  # Convert words to lowercase before adding to set
    if len(unique_words) > 500:
        unique_words = set(list(unique_words)[:500])
    unique_text = ' '.join(unique_words)
    return unique_text

def process_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        input_text = file.read()
    unique_text = remove_duplicates(input_text)
    output_file = os.path.splitext(input_file)[0] + '_unique.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(unique_text)
    print("Duplicates removed and unique words written to '{}'.".format(output_file))

input_folder = 'Lifestyle and Hobbies data'
input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.txt')]
for input_file in input_files:
    process_file(input_file)
