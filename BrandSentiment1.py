import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Add title and Write in 
st.header('Sentiment and Insights from Product Reviews: Exploratory Data Analysis')
st.write("Here's an overview of our LEGO dataset. In this dataset, we have information including lego set name, title in the lego set, reviews and etc.")

df = pd.read_csv("./DataFrames/lego_review_df_sentiment.csv")


st.dataframe(df.head(3))

st.header("😀Sentiment Analysis😒")

st.caption("""
Sentiment analysis is a natural language processing (NLP) technique used to determine whether text is positive, negative or neutral. Sentiment analysis is also known as “opinion mining” or “emotion artificial intelligence”. You can use it in e-commerce, politics, marketing, advertising, market research for example.
""")

st.subheader("Here are the TOP 10 LEGO sets with the most positive reviews😀")

st.table(df.groupby("set_name").sentiment.mean().reset_index()\
.round(2).sort_values("sentiment", ascending=False)\
.assign(Average_sentiment=lambda x: x.pop("sentiment").apply(lambda y: "%.2f" % y)).head(10))

st.subheader("Here are the TOP 10 LEGO sets with the most negative reviews😒")

st.table(df.groupby("set_name").sentiment.mean().reset_index()\
.round(2).sort_values("sentiment", ascending=False)\
.assign(Average_sentiment=lambda x: x.pop("sentiment").apply(lambda y: "%.2f" % y)).tail(10))



df['comment'] = df['comment'].astype(str)
def to_sentiment(rating):
    rating = int(rating)
    if rating <=2:
        return "negative"
    elif rating == 3:
        return "neutral" 
    else:
        return "positive"
df['sentiment'] = df.sentiment.apply(to_sentiment)


st.sidebar.markdown("### Number of Reviews")
select = st.sidebar.selectbox('Visualization Type',['Histogram','PieChart'])



sentiment_count = df['sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'Reviews':sentiment_count.values})

if st.sidebar.checkbox('Show',False,key='0'):
    st.markdown("### No. of Product Reviews by sentiments ")
    if select=='Histogram':
        fig = px.bar(sentiment_count,x='Sentiments',y='Reviews',color='Reviews',height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count,values='Reviews',names='Sentiments')
        st.plotly_chart(fig)

st.sidebar.subheader("Total number of comments for each lego set")
each_set = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
set_sentiment_count = df.groupby('set_name')['sentiment'].count().sort_values(ascending=False).iloc[:20]
set_sentiment_count = pd.DataFrame({'lego set':set_sentiment_count.index, 'Reviews':set_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True, key='2'):
    if each_set == 'Bar plot':
        st.subheader("Total number of comments for each lego set")
        fig_1 = px.bar(set_sentiment_count, x='lego set', y='Reviews', color='Reviews', height=500)
        st.plotly_chart(fig_1)
    if each_set == 'Pie chart':
        st.subheader("Total number of tweets for each lego set")
        fig_2 = px.pie(set_sentiment_count, values='Reviews', names='lego set')
        st.plotly_chart(fig_2)

#Word cloud
st.sidebar.subheader("Word Cloud")
word_sentiment = st.sidebar.radio("Which Sentiment to Display?", tuple(pd.unique(df["sentiment"])))

if st.sidebar.checkbox("Show", False, key="6"):
    st.subheader(f"Word Cloud for {word_sentiment.capitalize()} Sentiment")
    df = df[df["sentiment"]==word_sentiment]
    words = " ".join(df["comment"])
    processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=600, height=400).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

st.set_option('deprecation.showPyplotGlobalUse', False)