import collections
from collections.abc import Iterable
from typing import cast
from tensorflow import keras
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Clean Data
mails = pd.read_csv("enron_spam_data.csv")

mails.drop(labels=["Message ID", "Date"], axis=1, inplace=True)

mails.dropna(subset=["Message"], inplace=True)

mails.rename(
    columns={"Subject": "subject", "Message": "message", "Spam/Ham": "spam"},
    inplace=True,
)

mails["spam"] = mails["spam"].map(
    lambda spam_ham: True if spam_ham == "spam" else False
)

# Split Data
print(mails.head(5))
print()
print(mails["spam"].value_counts())

all_mails_count = mails["spam"].count()
print()
print("Counts:")
print("Total Count:", all_mails_count)

mails_split = train_test_split(mails, test_size=0.2, random_state=123, shuffle=True)
train_data, test_data = cast(tuple[pd.DataFrame, pd.DataFrame], mails_split)
train_data.reset_index(inplace=True, drop=True)
test_data.reset_index(inplace=True, drop=True)

print("Training Data:", train_data["spam"].count())
print("Test Data:", test_data["spam"].count())
print()

# Count words
spam_messages = set(train_data[train_data["spam"] == True]["message"])
ham_messages = set(train_data[train_data["spam"] == False]["message"])


def bar_chart_words(words, top=15, messages_type="", color="orange"):
    top_spam = np.array(sorted(words.items(), key=lambda x: -x[1]))[:top]
    top_words = top_spam[::-1, 0]
    top_words_count = [int(i) for i in top_spam[::-1, 1]]
    # aesthetics
    if messages_type:
        messages_type = messages_type + " "

    plt.title(f"Top {top} most common words in {messages_type}messages")
    plt.xlabel(f"Number of words")
    plt.barh(top_words, top_words_count, color=color)
    plt.show()


def wordcloud_words(words, max_words=15):
    wc = WordCloud(
        width=1024, height=1024, max_words=max_words
    ).generate_from_frequencies(words)
    plt.figure(figsize=(8, 6), facecolor="k")
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


words = pd.read_csv("words.csv", encoding="UTF-8")
wordlist = set(words["words"])
print("NLTK Downloads:")
nltk.download("stopwords")
nltk.download("punkt_tab")
print()
stop_words = set(stopwords.words("english"))


def process_message(message: str) -> list[str]:
    words = message.lower()  # lowercase
    words = word_tokenize(words)  # tokenization gets rid of punctuation and
    words = [word for word in words if len(word) > 1]  # non absurd words
    words = [word for word in words if word not in stop_words]  # non stop words
    words = [word for word in words if word in wordlist]  # english words
    # stemming basically gets rid of all the english tenses
    # leaving with a single meaning word without any suffix variations
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  # stemming
    return words


def count_processed_words(
    data: Iterable[str] | Iterable[Iterable[str]], pre_processed=False
) -> dict[str, int]:
    counter = collections.OrderedDict()
    for message in data:
        words = message if pre_processed else process_message(cast(str, message))

        for word in set(words):
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    return counter


# spam_words = count_processed_words(spam_messages)
# ham_words = count_processed_words(ham_messages)

# Bar charts and Word Clouds
# wordcloud_words(spam_words)
# wordcloud_words(ham_words)

# bar_chart_words(spam_words, messages_type="spam", color="orange")
# bar_chart_words(ham_words, messages_type="ham", color="green")


# Processing messages for empty strings
print("Processing messages...")
processed_spam_messages = [process_message(i) for i in spam_messages]
processed_spam_messages = [i for i in processed_spam_messages if len(i) >= 1]
processed_ham_messages = [process_message(i) for i in ham_messages]
processed_ham_messages = [i for i in processed_ham_messages if len(i) >= 1]

print("Counting words...")
spam_words = count_processed_words(processed_spam_messages, pre_processed=True)
ham_words = count_processed_words(processed_ham_messages, pre_processed=True)

print("Counting all words...")
processed_all_messages = processed_spam_messages + processed_ham_messages
all_words = count_processed_words(processed_all_messages, pre_processed=True)

print()


def spam(message, s=1, p=0.5, percentage=False, pre_processed=False):
    """
    message - needs to be a non-empty string value for valid result
    s - the strength we give to background information about incoming spam, default is 1
    p - the probability of any incoming message to be spam, default is 0.5
    percentage - returns result as boolean or a percentage, default is True
    """
    n = 0
    spam_freq = 0
    ham_freq = 0
    message_words = message if pre_processed else process_message(message)
    for word in message_words:
        if word in spam_words.keys():
            # count of spam messages containing the word / count of all messages containing the word
            spam_freq = spam_words[word] / all_words[word]

        if word in ham_words.keys():
            # count of ham messages containing the word / count of all messages containing the word
            ham_freq = ham_words[word] / all_words[word]

        # if word is not in trained dataset we ignore it
        if not (spam_freq + ham_freq) == 0 and word in all_words.keys():
            spaminess_of_word = (spam_freq) / (spam_freq + ham_freq)
            corr_spaminess = (s * p + all_words[word] * spaminess_of_word) / (
                s + all_words[word]
            )
            n += np.log(1 - corr_spaminess) - np.log(corr_spaminess)

    spam_result = 1 / (1 + np.e**n)

    if percentage:
        print(f"Spam probability: {spam_result * 100:.2f}%")
    elif spam_result > 0.5:
        return True
    else:
        return False


def test(spam_test, ham_test, s=1, p=0.5, details=False, pre_processed=False):
    """
    spam_test - list of spam messages to be tested
    ham_test - list of ham messages to be tested
    details - displays additional information
    """
    spam_count = 0
    ham_count = 0
    for message in spam_test:
        if spam(message, s, p, pre_processed=pre_processed):
            spam_count += 1
        else:
            ham_count += 1

    true_positive = spam_count
    false_negative = ham_count

    spam_count = 0
    ham_count = 0
    for message in ham_test:
        if spam(message, s, p, pre_processed=pre_processed):
            spam_count += 1
        else:
            ham_count += 1

    false_positive = spam_count
    true_negative = ham_count

    # How many selected messages are spam?
    spam_precision = true_positive / (true_positive + false_positive)

    # How many spam messages are selected?
    spam_recall = true_positive / (true_positive + false_negative)

    # Harmonic mean between precision and recall.
    spam_fscore = 2 * (spam_precision * spam_recall) / (spam_precision + spam_recall)

    # How many selected messages are ham?
    ham_precision = true_negative / (true_negative + false_negative)

    # How many ham messages are selected?
    ham_recall = true_negative / (true_negative + false_positive)

    # Harmonic mean between precision and recall.
    ham_fscore = 2 * (ham_precision * ham_recall) / (ham_precision + ham_recall)

    # If the data was ballanced.
    # accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    # For unballanced data.
    ballanced_accuracy = (spam_recall + ham_recall) / 2

    if details:
        print("True Positive: ", true_positive)
        print("False Negative:", false_negative)
        print("True Negative: ", true_negative)
        print(f"False Positive: {false_positive}\n")

        print(f"Spam precision: {spam_precision * 100:.2f}%")
        print(f"Spam recall: {spam_recall * 100:.2f}%")
        print(f"Spam F-score: {spam_fscore * 100:.2f}%\n")

        print(f"Ham precision: {ham_precision * 100:.2f}%")
        print(f"Ham recall: {ham_recall * 100:.2f}%")
        print(f"Ham F-score: {ham_fscore * 100:.2f}%\n")

    print(f"Accuracy: {ballanced_accuracy * 100:.2f}%\n")


print("Testing...")
print()

test_spam_messages = set(test_data[test_data["spam"] == True]["message"])
test_spam_messages = [process_message(i) for i in test_spam_messages]
test_spam_messages = [i for i in test_spam_messages if len(i) >= 1]

test_ham_messages = set(test_data[test_data["spam"] == False]["message"])
test_ham_messages = [process_message(i) for i in test_ham_messages]
test_ham_messages = [i for i in test_ham_messages if len(i) >= 1]

test(
    spam_test=test_spam_messages,
    ham_test=test_ham_messages,
    details=True,
    pre_processed=True,
)
