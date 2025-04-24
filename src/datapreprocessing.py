import pandas as pd
import re
import string
import emoji

df = pd.read_csv('data\cyberbullying_tweets.csv')
df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})

def strip_emoji(text):
    return emoji.get_emoji_regexp().sub("", text)

def strip_all_entities(text):
    text = re.sub(r'\r|\n', ' ', text.lower()) 
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split())
    text = ' '.join(word for word in text.split())
    return text

def clean_hashtags(tweet):
    new_tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip() 
    new_tweet = re.sub(r'#([\w-]+)', r'\1', new_tweet).strip()

    return new_tweet

def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

def replace_elongated_words(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

def remove_repeated_punctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_url_shorteners(text):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', text)

def remove_spaces_tweets(tweet):
    return tweet.strip()

def remove_short_tweets(tweet, min_words=2):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""

def clean_tweet(tweet):
    tweet = strip_emoji(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = ' '.join(tweet.split())
    return tweet

df['text_clean'] = [clean_tweet(tweet) for tweet in df['text']]
df.drop_duplicates("text_clean", inplace=True)
df = df[df["sentiment"]!="other_cyberbullying"]
df['text_len'] = [len(text.split()) for text in df.text_clean]

df = df[df["text_len"]<=31]
df = df[df["text_len"]>0]

def select_and_assign(df, categories, n_samples):
    indices = []
    for category in categories:
        selected_idx = df[df['sentiment'] == category].sample(n=n_samples, random_state=1).index
        indices.extend(selected_idx)
    return df.loc[indices]

cyberbullying_df = select_and_assign(df, ['religion', 'age', 'ethnicity', 'gender', 'not_cyberbullying'], 800)

not_cyberbullying_df = df[df['sentiment'] == 'not_cyberbullying'].sample(n=3000, random_state=1)

result_df = pd.concat([cyberbullying_df, not_cyberbullying_df], ignore_index=True)

df = cyberbullying_df
df['sentiment'] = df['sentiment'].replace({'cyberbullying':1,'not_cyberbullying':0})
df['sentiment'] = df['sentiment'].replace({'religion':0,'age':1,'ethnicity':2,'gender':3,'not_cyberbullying':4})
df.to_csv('preprocessed_data.csv', index=False)