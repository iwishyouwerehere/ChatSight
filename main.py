import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import nltk

# Check if VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to read and analyze each HTML file
def read_and_analyze(file_path):
    columns = ['Date', 'Sentiment', 'Category', 'Text', 'Word_Count']
    df = pd.DataFrame(columns=columns)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            messages = soup.find_all('div', {'class': 'message default clearfix'})

            for message in messages:
                date = message.find('div', {'class': 'pull_right date details'})['title']
                text_div = message.find('div', {'class': 'text'})

                if text_div:
                    text = text_div.text
                    sentiment = sia.polarity_scores(text)['compound']
                    word_count = len(text.split())

                    category = 'Neutral'
                    if sentiment > 0.05:
                        category = 'Positive'
                    elif sentiment < -0.05:
                        category = 'Negative'

                    new_row = pd.DataFrame({'Date': [date], 'Sentiment': [sentiment], 'Category': [category], 'Text': [text], 'Word_Count': [word_count]})
                    df = pd.concat([df, new_row]).reset_index(drop=True)
    except Exception as e:
        print(f"An error occurred: {e}")
    return df

# Function to identify the topic of conversation
def identify_topic(df):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(df['Text'])
    lda = LDA(n_components=1)
    lda.fit(count_data)
    words = count_vectorizer.get_feature_names_out()
    topic = lda.components_[0]
    chosen_topic = words[topic.argmax()]
    if any(char.isdigit() for char in chosen_topic):
        return "N/A"
    return chosen_topic

# Function to identify the most used word of the month
def most_used_word(df):
    all_words = ' '.join(df['Text']).split()
    filtered_words = [word for word in all_words if len(word) > 3]
    counter = Counter(filtered_words)
    return counter.most_common(1)[0]

# Function to update the plot
def update(frame, files, ax1, ax2, ax3, ax4, ax5):
    df = read_and_analyze(files[frame])
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Convert 'Date' to datetime format and extract only the date part
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S UTC%z', dayfirst=True).dt.date

    # Sentiment Categories Count
    sentiment_count = df.groupby(['Date', 'Category']).size().unstack(fill_value=0)
    sentiment_count.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Sentiment Categories Count')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Count')

    # Average Sentiment Score
    avg_sentiment = df.groupby('Date')['Sentiment'].mean()
    avg_sentiment.plot(kind='line', ax=ax2)
    ax2.set_title('Average Sentiment Score')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Average Sentiment')

    # Word Count
    word_count = df.groupby('Date')['Word_Count'].sum()
    word_count.plot(kind='bar', ax=ax3)
    ax3.set_title('Word Count')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Total Words')

     # Most Used Word of the Month
    most_word = most_used_word(df)
    ax4.clear()
    ax4.bar(most_word[0], most_word[1])
    ax4.set_title('Most Used Word of the Month')
    ax4.set_xlabel('Word')
    ax4.set_ylabel('Frequency')

    # Topic of Conversation Every Day
    topic = identify_topic(df)
    ax5.clear()
    ax5.text(0.5, 0.5, topic, ha='center', va='center')
    ax5.set_title('Topic of Conversation')
    ax5.axis('off')

# Initialize plot
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 10))

# Get sorted list of files
files = sorted(glob.glob("C:/Users/DonElladio/Downloads/Telegram Desktop/ChatExport_2023-10-10/messages*.html"))

# Initialize animation
ani = FuncAnimation(fig, update, frames=range(len(files)), fargs=(files, ax1, ax2, ax3, ax4, ax5), repeat=False)

plt.tight_layout()
plt.show()
