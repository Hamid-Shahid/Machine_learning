import os
import nltk
import string
import itertools
from grakel import WeisfeilerLehman, graph_from_networkx
from collections import defaultdict
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

# Function to create directed graph from text
def text_to_graph(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Preprocess text
    tokens = preprocess_text(text)
    # Create directed graph
    G = nx.DiGraph()
    
    prev_token = None
    for token in tokens:
        if prev_token:
            if not G.has_edge(prev_token, token):
                G.add_edge(prev_token, token, weight=1)
            else:
                G[prev_token][token]['weight'] += 1
        prev_token = token
    return G

def get_score(g1, g2):
        common_graph = nx.Graph()
        for n1, n2 in g2.edges():
            if g1.has_edge(n1, n2):
                common_graph.add_edge(n1, n2)
        components = list(nx.connected_components(common_graph))
        return sum([len(i) for i in components]) / min(g1.number_of_nodes(), g2.number_of_nodes())


def knn_predict(train_graphs, test_graph):
        scores = []
        for category, graphs in train_graphs.items():
            for train_graph in graphs:
                score = get_score(test_graph, train_graph)
                scores.append((category, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_35 = [category for category, _ in scores[:35]]
        majority_label = Counter(top_35).most_common(1)[0][0]
        return majority_label

def knn_classification(train_graphs, test_graphs):
        y_pred = []
        y_test = []
        for category, graphs in test_graphs.items():
            for test_graph in graphs:
                prediction = knn_predict(train_graphs, test_graph)
                y_pred.append(prediction)
                y_test.append(category)
        return y_pred, y_test


def evaluate_classification(true_labels, predicted_labels):
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        return accuracy, precision, recall, f1

def plot_confusion_matrix(true_labels, predicted_labels, classes):
        cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

def main():
    graphs = []
    test_graphs=[]
    train_graphs_dic = defaultdict()
    test_graphs_dic=defaultdict()
    directories = ['Lifestyle and Hobbies','Sports','Business and Finance']
    for directory in directories:   
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                graph = text_to_graph(file_path)
                graphs.append(graph)
            # You can save or visualize the graph here as needed
                # print("Graph created for", filename)
        train_graphs_dic[directory]=graphs
        graphs = []
        

    directories = ['Lifestyle and Hobbies test','Sports test','Business and Finance test']
    for directory in directories:   
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                test_graph = text_to_graph(file_path)
                test_graphs.append(test_graph)
            # You can save or visualize the graph here as needed
                # print("Graph created for", filename)

        test_graphs_dic[directory[:len(directory)-5]]=test_graphs
        test_graphs = []
    pred,true=knn_classification(train_graphs_dic,test_graphs_dic)
    print(evaluate_classification(true,pred))
    plot_confusion_matrix(true,pred,['Lifestyle and Hobbies','Sports','Business and Finance'])
if __name__ == "__main__":
    main()