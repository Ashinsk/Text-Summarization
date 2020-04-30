from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


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

def create_word_frequency_table(words: list) -> dict:
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
    """
    print('Finding average score')

    sum_values = 0
    for sentence in sent_value:
        sum_values += sent_value[sentence]

    average = int(sum_values / len(sent_value))

    return average


def generate_summary(sentences: list, sent_value: dict, avg: int) -> str:
    """
    Generate a sentence for sentence score greater than average.
    """
    print('Generating summary')

    sentence_count = 0
    summary = ''

    for sent in sentences:
        if sent[:15] in sent_value and sent_value[sent[:15]] >= avg:
            summary += sent + " "
            sentence_count += 1

    return summary


def main():
    text = "Your text goes here."

    sentences = sent_tokenize(text.strip())
    print('Sentences',sentences)
    
    clean_words = text_preprocessing(sentences)
    print('Clean Words',clean_words)

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