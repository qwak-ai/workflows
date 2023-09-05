from pandas import DataFrame
from qwak import QwakClient

#utility function to count syllables in a word
def syllable_count(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


# calls the OfflineClient to retrieve the feature set data frame from Qwak
def get_data_from_feature_store(query) -> DataFrame:

    client = QwakClient()

    return client.run_analytics_query(query=query)
