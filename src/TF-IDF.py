from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import math

ps = PorterStemmer()


def text_preprocessing(sentences):
    """
    Pre processing text to remove unnecessary words.
    """
    print('Preprocessing text')

    stop_words = set(stopwords.words('english'))

    clean_words = None
    for sent in sentences:
        words = word_tokenize(sent)
        words = [ps.stem(word.lower()) for word in words if word.isalnum()]
        clean_words = [word for word in words if word not in stop_words]

    return clean_words


def create_tf_matrix(sentences: list) -> dict:
    """
    Here document refers to a sentence.
    TF(t) = (Number of times the term t appears in a document) / (Total number of terms in the document)
    """
    print('Creating tf matrix.')

    tf_matrix = {}

    for sentence in sentences:
        tf_table = {}

        words_count = len(sentence)
        clean_words = text_preprocessing([sentence])

        # Determining frequency of words in the sentence
        word_freq = {}
        for word in clean_words:
            word_freq[word] = (word_freq[word] + 1) if word in word_freq else 1

        # Calculating tf of the words in the sentence
        for word, count in word_freq.items():
            tf_table[word] = count / words_count

        tf_matrix[sentence[:15]] = tf_table

    return tf_matrix


def create_idf_matrix(sentences: list) -> dict:
    """
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    """
    print('Creating idf matrix.')

    idf_matrix = {}

    documents_count = len(sentences)
    sentence_word_table = {}

    # Getting words in the sentence
    for sentence in sentences:
        clean_words = text_preprocessing([sentence])
        sentence_word_table[sentence[:15]] = clean_words

    # Determining word count table with the count of sentences which contains the word.
    word_in_docs = {}
    for sent, words in sentence_word_table.items():
        for word in words:
            word_in_docs[word] = (word_in_docs[word] + 1) if word in word_in_docs else 1

    # Determining idf of the words in the sentence.
    for sent, words in sentence_word_table.items():
        idf_table = {}
        for word in words:
            idf_table[word] = math.log10(documents_count / float(word_in_docs[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def create_tf_idf_matrix(tf_matrix, idf_matrix) -> dict:
    """
    Create a tf-idf matrix which is multiplication of tf * idf individual words
    """
    print('Calculating tf-idf of sentences.')

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def create_sentence_score_table(tf_idf_matrix) -> dict:
    """
    Determining average score of words of the sentence with its words tf-idf value.
    """
    print('Creating sentence score table.')

    sentence_value = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

            sentence_value[sent] = total_score_per_sentence / count_words_in_sentence

    return sentence_value


def find_average_score(sentence_value):
    """
    Calculate average value of a sentence form the sentence score table.
    """
    print('Finding average score')

    sum = 0
    for val in sentence_value:
        sum += sentence_value[val]

    average = sum / len(sentence_value)

    return average


def generate_summary(sentences, sentence_value, threshold):
    """
    Generate a sentence for sentence score greater than average.
    """
    print('Generating summary')

    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentence_value and sentence_value[sentence[:15]] >= threshold:
            summary += sentence + " "
            sentence_count += 1

    return summary


def main():
    text = "If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It’s a fact, if you don’t know what you want you’ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don’t figure out what you want, you’ll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it."

    sentences = sent_tokenize(text)
    print('Sentences', sentences)

    tf_matrix = create_tf_matrix(sentences)
    print('TF matrix', tf_matrix)

    idf_matrix = create_idf_matrix(sentences)
    print('IDF matrix',idf_matrix)

    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
    print('TF-IDF matrix', tf_idf_matrix)

    sentence_value = create_sentence_score_table(tf_idf_matrix)
    print('Sentence Scores', sentence_value)

    threshold = find_average_score(sentence_value)
    print('Threshold', threshold)

    summary = generate_summary(sentences, sentence_value, threshold)

    print('\nOriginal document\n',text,end='\n'*2)
    print('Summary\n',summary)


if __name__ == '__main__':
    main()
