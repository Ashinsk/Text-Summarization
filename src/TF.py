from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


ps = PorterStemmer()


def text_preprocessing(sentences: list) -> list:
    """
    Pre processing text to remove unnecessary words.
    """
    print('Preprocessing text')

    stop_words = set(stopwords.words('english'))

    clean_words = []
    for sent in sentences:
        # Tokenizing words.
        words = word_tokenize(sent.lower())
        # Removing non alphabetic and numeric words.
        words = [ps.stem(word) for word in words if word.isalnum()]
        # Removing stopwords
        clean_words += [word for word in words if word not in stop_words]

    return clean_words


def create_word_frequency_table(words: list) -> dict:
    """
    Creating word frequency table which contains frequency of each word used in the text.
    """
    print('Creating word frequency table')

    freq_table = dict()

    for word in words:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    return freq_table


def create_sentence_score_table(sentences: list, freq_table: dict) -> dict:
    """
    Creating a dictionary to keep the score of each sentence.
    Sore is the sum of frequency of words used in the sentence.
    """
    print('Creating sentence score table')

    sent_value = dict()
    for sentence in sentences:
        for word, freq in freq_table.items():
            if ps.stem(word) in sentence.lower():
                if sentence[:15] in sent_value:
                    sent_value[sentence[:15]] += freq
                else:
                    sent_value[sentence[:15]] = freq

    return sent_value


def find_average_score(sent_value: dict) -> int:
    """
    Calculate average value of a sentence form the original text.
    Average value is the sum value divided by the number of sentences in the text.
    """
    print('Finding average score')

    sum_values = 0
    for sentence in sent_value:
        sum_values += sent_value[sentence]

    average = int(sum_values / len(sent_value))

    return average


def generate_summary(sentences: list, sent_value: dict, avg: int) -> str:
    """
    To generate the summary, extracting the sentences having sentence score greater than or equal to average.
    """
    print('Generating summary')

    summary = ''
    for sent in sentences:
        if sent[:15] in sent_value and sent_value[sent[:15]] >= avg:
            summary += sent + " "

    return summary


def main():
    text = "Your text goes here."

    text = ""

    with open('../File_1_en.txt', "r+") as f:
        for line in f:
            text += line

    sentences = sent_tokenize(text.strip())
    print('Sentences',len(sentences),sentences)
    
    clean_words = text_preprocessing(sentences)
    print('Clean Words',len(clean_words),clean_words)

    freq_table = create_word_frequency_table(clean_words)
    print('Frequency Table',freq_table)

    sent_values = create_sentence_score_table(sentences, freq_table)
    print('Sentence values',sent_values)

    average = find_average_score(sent_values)
    print('Average',average)

    summary = generate_summary(sentences, sent_values, average)

    print('\nOriginal document\n',text,end='\n'*2)
    print('Summary\n',summary)


if __name__ == "__main__":
    main()
