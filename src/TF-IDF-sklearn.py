from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd


def create_sentence_score_table(sentences, tfidf_vector) -> dict:
    """
    Determining average score of words of the sentence with its words tf-idf value.
    """
    # print('Creating sentence score table.')

    sentence_value = {}

    for sentence, sent_vector in zip(sentences, tfidf_vector):
        df = pd.DataFrame(sent_vector.T.todense(), columns=['tfidf'])
        total_score_per_sentence = df.sum()['tfidf']
        count_words_in_sentence = len(sentences)
        smoothing = 1
        sentence_value[sentence[:15]] = (total_score_per_sentence + smoothing) / (count_words_in_sentence + smoothing)

    return sentence_value


def find_average_score(sentence_value):
    """
    Calculate average value of a sentence form the sentence score table.
    """
    # print('Finding average score')

    sum = 0
    for val in sentence_value:
        sum += sentence_value[val]

    average = sum / len(sentence_value)

    return average


def generate_summary(sentences, sentence_value, threshold):
    """
    Generate a sentence for sentence score greater than average.
    """
    # print('Generating summary')

    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentence_value and sentence_value[sentence[:15]] >= threshold:
            summary += sentence + " "
            sentence_count += 1

    return summary


def main():
    text = ""
    with open('src/File_1_en', "r+") as f:
        for line in f:
            text += line

    sentences = sent_tokenize(text)  # docs
    # print('Sentences', sentences)

    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(sentences)
    # print('Word count vector', word_count_vector.shape, word_count_vector)

    feature_names = cv.get_feature_names()
    # print('Feature names',len(feature_names),feature_names)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=['idf_weights'])
    df_idf.sort_values(by=['idf_weights'])
    # print(df_idf.head())
    # print(df_idf.tail())

    count_vector = cv.transform(sentences)
    tfidf_vector = tfidf_transformer.transform(count_vector)

    first_document_vector = tfidf_vector[0]
    df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=['tfidf'])
    df.sort_values(by=['tfidf'], ascending=False)

    sentence_value = create_sentence_score_table(sentences, tfidf_vector)
    # print('Sentence Scores', sentence_value)

    threshold = find_average_score(sentence_value)
    # print('Threshold', threshold)

    summary = generate_summary(sentences, sentence_value, threshold)

    # print('\nOriginal document\n',text,end='\n'*2)
    print('Summary\n', summary)

    print()
    print(f'Original {len(sent_tokenize(text))} sentences, Summarized {len(sent_tokenize(summary))} sentences')


if __name__ == '__main__':
    main()
