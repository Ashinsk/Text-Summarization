import networkx as nx
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

wl = WordNetLemmatizer()


def extract_word_vectors() -> dict:
    """
    Extracting word embeddings. These are the n vector representation of words.
    """
    print('Extracting word vectors')

    word_embeddings = {}
    # Here we use glove word embeddings of 100 dimension
    f = open('../../Files/glove.6B/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs

    f.close()
    return word_embeddings


def text_preprocessing(sentences: list) -> list:
    """
    Pre processing text to remove unnecessary words.
    """
    print('Preprocessing text')

    stop_words = set(stopwords.words('english'))

    clean_words = None
    for sent in sentences:
        words = word_tokenize(sent)
        words = [wl.lemmatize(word.lower()) for word in words if word.isalnum()]
        clean_words = [word for word in words if word not in stop_words]

    return clean_words


def sentence_vector_representation(sentences: list, word_embeddings: dict) -> list:
    """
    Creating sentence vectors from word embeddings.
    """
    print('Sentence embedding vector representations')

    sentence_vectors = []
    for sent in sentences:
        clean_words = text_preprocessing([sent])
        # Averaging the sum of word embeddings of the sentence to get sentence embedding vector
        v = sum([word_embeddings.get(word, np.zeros(100, )) for word in clean_words]) / (len(clean_words) + 0.001)
        sentence_vectors.append(v)

    return sentence_vectors


def create_similarity_matrix(sentences: list, sentence_vectors: list) -> np.ndarray:
    """
    Using cosine similarity, generate similarity matrix.
    """
    print('Creating similarity matrix')

    # Defining a zero matrix of dimension n * n
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                # Replacing array value with similarity value.
                # Not replacing the diagonal values because it represents similarity with its own sentence.
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    return sim_mat


def determine_sentence_rank(sentences: list, sim_mat: np.ndarray):
    """
    Determining sentence rank using Page Rank algorithm.
    """
    print('Determining sentence ranks')
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted([(scores[i], s[:15]) for i, s in enumerate(sentences)], reverse=True)
    return ranked_sentences


def generate_summary(sentences: list, ranked_sentences: list):
    """
    Generate a sentence for sentence score greater than average.
    """
    print('Generating summary')

    # Get top 1/3 th ranked sentences
    top_ranked_sentences = ranked_sentences[:int(len(sentences) / 3)]

    sentence_count = 0
    summary = ''

    for i in sentences:
        for j in top_ranked_sentences:
            if i[:15] == j[1]:
                summary += i + ' '
                sentence_count += 1
                break

    return summary


def main():
    text = "Your text goes here."

    sentences = sent_tokenize(text.strip())
    print('Sentences',sentences)

    word_embeddings = extract_word_vectors()
    print('Word embeddings',len(word_embeddings))

    sentence_vectors = sentence_vector_representation(sentences, word_embeddings)
    print('Sentence vectors', len(sentence_vectors), sentence_vectors)

    sim_mat = create_similarity_matrix(sentences, sentence_vectors)
    print('Similarity matrix', sim_mat.shape, sim_mat)

    ranked_sentences = determine_sentence_rank(sentences, sim_mat)
    print('Ranked sentences', ranked_sentences)

    summary = generate_summary(sentences, ranked_sentences)

    print('\nOriginal document\n',text,end='\n'*2)
    print('Summary\n',summary)


if __name__ == "__main__":
    main()
