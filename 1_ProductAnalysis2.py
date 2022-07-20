import streamlit as st
from streamlit_chat import message
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import plotly.express as px
from wordcloud import WordCloud
from PIL import Image
import gensim
from gensim.parsing.preprocessing import STOPWORDS

# ---- HEADER ---

st.set_page_config(page_title="Extracting Sentiment and Insights from Product Reviews",
                   layout="wide")

t1, t2 = st.columns((0.07,1))
t1.image('./Images/LEGO_logo.svg.png', width = 100)
t2.title("Extracting Sentiment and Insights from Product Reviews")


# ---- FUNCTIONS ---

@st.cache
def color_sentiment(sentiment):
    colors = {
        "POS": "yellow",
        "NEU": "green",
        "NEG": "red"
    }
    return f"background-color: {colors[sentiment]}"

@st.cache
def get_allItems(df, allTriple):
	return df[df.set_name.isin(allTriple.item.unique())].set_name.unique()

@st.cache
def get_triples_by_product(selected_products, selected_emotions=['POS','NEU','NEG']):
	return allTriple[allTriple.item.isin(selected_products) & allTriple.sentiment.isin(selected_emotions)].reset_index(drop=True)
	#return allTriple[allTriple.item.isin(selected_products)].reset_index(drop=True)

@st.cache
def get_df_by_product(selected_products, selected_emotions=['POS','NEU','NEG']):
	return df[df.set_name.isin(selected_products) & df.BERT_SentLabel.isin(selected_emotions)].reset_index(drop=True)

@st.cache(suppress_st_warning=True)
def plot_pos_wordcloud(pos_keywords, out = 'target'):
    if out == 'target':
        cm = ' '.join(list(map(str, pos_keywords)))
        mask = np.array(Image.open('./Images/upvote.png'))
        wcP = WordCloud(stopwords=stop_words, width=3000, height=2000, random_state=1, collocations=False,
                        background_color="white", colormap='rainbow', mask=mask).generate(cm)
        return wcP
    else:
        filtered_words = [word for word in ' '.join(pos_keywords).split() if word not in stop_words]
        top_word = collections.Counter(filtered_words).most_common(1)[0][0]
        temp = triple_selected[triple_selected.sentiment == 'POS'][[top_word in w for w in triple_selected[triple_selected.sentiment == 'POS'].target]]
        cm = ' '.join(list(map(str, temp.opinion)))
        wcP = WordCloud(stopwords=stop_words, collocations=False, background_color="white").generate(cm)
        return wcP, top_word


@st.cache(suppress_st_warning=True)
def plot_neg_wordcloud(neg_keywords, out = 'target'):
    if out == 'target':
        cm = ' '.join(list(map(str, neg_keywords)))
        mask = np.array(Image.open('./Images/downvote.png'))
        wcN = WordCloud(stopwords=stop_words, width=2000, height=1500, random_state=1, collocations=False,
                        background_color="white", colormap='rainbow', mask=mask).generate(cm)
        return wcN
    else:
        filtered_words = [word for word in ' '.join(neg_keywords).split() if word not in stop_words]
        top_word = collections.Counter(filtered_words).most_common(1)[0][0]
        temp = triple_selected[triple_selected.sentiment == 'NEG'][[top_word in w for w in triple_selected[triple_selected.sentiment == 'NEG'].target]]
        cm = ' '.join(list(map(str, temp.opinion)))
        wcN = WordCloud(stopwords=stop_words, collocations=False, background_color="white").generate(cm)
        return wcN, top_word

@st.cache(suppress_st_warning=True)
def plot_neu_wordcloud(neu_keywords, out = 'target'):
    if out == 'target':
        cm = ' '.join(list(map(str, neu_keywords)))
        wc = WordCloud(stopwords=stop_words, width=3000, height=2000, collocations=False,
                       background_color="white", colormap='rainbow').generate(cm)
        return wc
    else:
        filtered_words = [word for word in ' '.join(neu_keywords).split() if word not in stop_words]
        top_word = collections.Counter(filtered_words).most_common(1)[0][0]
        temp = triple_selected[triple_selected.sentiment == 'NEU'][[top_word in w for w in triple_selected[triple_selected.sentiment == 'NEU'].target]]
        cm = ' '.join(list(map(str, temp.opinion)))
        wc = WordCloud(stopwords=stop_words, collocations=False, background_color="white").generate(cm)
        return wc, top_word



# ---- IMPORT DATA ---

df = pd.read_csv("./DataFrames/df_running.csv")
allTriple = pd.read_csv("./DataFrames/allTriple0_5200.csv")



# ---- SIDEBAR ---
#st.sidebar.header("Please Make Selections Here:")

st.markdown('### 1. Select A Product:')
selected_products = [st.selectbox(
    label = 'Products sorted by counts of reviews:',
    options=get_allItems(df, allTriple)
)]

df_selected = get_df_by_product(selected_products)
triple_selected = get_triples_by_product(selected_products)

# ---- MAINPAGE ---

Rating = round(df_selected.overallRating.mean(), 2)
Price = round(df_selected.Price.mean(), 2)
valueForMoney = round(df_selected.valueForMoney.mean(), 2)
overallPlayExperience = round(df_selected.overallPlayExperience.mean(), 2)
SentScore = round(df_selected.sentiment.mean(), 2)
NumTriples = triple_selected.reviewID.count()
NumReviews = df_selected.comment.shape[0]
#st.write(Rating, valueForMoney, overallPlayExperience, NumReviews, n, SentScore)

col1, col15, col2 , col3 , col4 , col5 = st.columns(6)

with col1:
    st.markdown("##### Product Name")
    st.markdown(f"##### 🧱{selected_products[0]}")

with col15:
    st.markdown("##### Listing Price")
    st.markdown(f"##### 🏷️${Price}")

with col2:
    st.markdown("##### Customers' Rating")
    st.markdown(f"##### :star:{Rating}")

with col3:
    st.markdown("##### BERT Sentiment Score")
    st.markdown(f"##### :smiley:{SentScore}")

with col4:
    st.markdown("##### Number of Reviews")
    st.markdown(f"##### :speech_balloon:{NumReviews}")

with col5:
    st.markdown("##### Number of Keywords")
    st.markdown(f"##### :bulb:{NumTriples}")

st.markdown('''---''')

# ---- Sentiment ---

st.markdown('### 2. Select the Sentiment(s):')
selected_emotions = st.multiselect(
    label = 'POS: Positive; NEU: Neutral; NEG: Negative',
    options= ['POS', 'NEU', 'NEG'],
    default = ['POS', 'NEU', 'NEG']
)

df_emo_selected = get_df_by_product(selected_products, selected_emotions)
triple_selected2 = get_triples_by_product(selected_products, selected_emotions)

st.markdown('#### Product Reviews')

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df_emo_selected.loc[:,['comment','sentiment','BERT_SentLabel']])

with col2:
    checked = st.checkbox(
        "Show detailed reviews and sentiment scores (0-1)",
        key="idI"
    )
    if checked:
        for i in range(min(2, NumReviews)):
            message(df_emo_selected.comment[i], key=f"rv{i}")
            message(str(df_emo_selected.sentiment[i]), is_user=True, key=f"sent{i}")

    checked2 = st.checkbox(
        "Show More (2-5)",
        key="idX"
    )
    if checked2 and NumReviews>2:
        for i in range(2, min(6, NumReviews)):
            message(df_emo_selected.comment[i], key=f"rv2{i}")
            message(str(df_emo_selected.sentiment[i]), is_user=True, key=f"sent2{i}")
        message("......")

st.markdown('''---''')

# ---- An example ---

st.markdown('### 3. Keywords Extraction (A Quick Demo)')

t1,t2,t3 = st.columns((0.07,1,0.07))
t2.image('./Images/Presentation1.png', use_column_width = True,clamp = True)


st.markdown('''---''')

# ---- Keywords Extraction ---

st.markdown('### 4. Insights Dashboard')

col1, col2 = st.columns(2)

with col1:
    st.markdown('#### Extracted Keywords')
    st.dataframe(triple_selected2.loc[:,['item','target','opinion','sentiment']].style.applymap(
        color_sentiment,
        subset=["sentiment"]
    ))

with col2:
    st.markdown('#### Distribution of Sentiment of Keywords')
    long_df = pd.DataFrame(triple_selected2.sentiment.value_counts()).reindex(['POS', 'NEU', 'NEG']).reset_index()
    long_df = long_df.rename(columns={'index': 'Sentiment', 'sentiment': 'Counts'})
    fig = px.bar(long_df, x="Sentiment", y="Counts", color="Sentiment")
    fig.update_layout(barmode='relative')
    fig

with col2:
    #st.markdown('#### Distribution of Sentiment of Keywords')
    long_df = pd.DataFrame(triple_selected2.sentiment.value_counts()).reindex(['POS', 'NEU', 'NEG']).reset_index()
    long_df = long_df.rename(columns={'index': 'Sentiment', 'sentiment': 'Counts'})
    fig = px.bar(long_df, x="Sentiment", y="Counts", color="Sentiment",
                 color_discrete_map={
                     "POS": "#17becf",
                     "NEU": "#2ca02c",
                     "NEG": "#d62728"}
                 )
    fig.update_layout(barmode='relative')
    #fig

# ---- Word Cloud ---


pos_keywords = [k for k in triple_selected[triple_selected.sentiment=='POS'].target]#.extend([k for k in allTriple[allTriple.sentiment=='POS'].opinion])
neg_keywords = [k for k in triple_selected[triple_selected.sentiment=='NEG'].target]#.extend([k for k in allTriple[allTriple.sentiment=='NEG'].opinion])
neu_keywords = [k for k in triple_selected[triple_selected.sentiment=='NEU'].target]#.extend([k for k in allTriple[allTriple.sentiment=='NEU'].opinion])
stop_words = ["lego", "legos", "set", "sets", "build", "building", "built", "piece", "pieces", "batman", "batmobile", "really", "completely"] + list(STOPWORDS) + selected_products[0].lower().split()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h4 style='text-align: center'>Positive Reviews Mentioned:</h4>", unsafe_allow_html=True)
    if pos_keywords:
        fig, ax = plt.subplots()
        ax.imshow(plot_pos_wordcloud(pos_keywords), aspect='auto')
        ax.axis('off')
        st.pyplot(fig)

with col2:
    st.markdown("<h4 style='text-align: center'>Negative Reviews Mentioned:</h4>", unsafe_allow_html=True)
    if neg_keywords:
        fig, ax = plt.subplots()
        ax.imshow(plot_neg_wordcloud(neg_keywords))
        ax.axis('off')
        st.pyplot(fig)

with col3:
    st.markdown("<h4 style='text-align: center'>Neutral Reviews Mentioned:</h4>", unsafe_allow_html=True)
    if neu_keywords:
        fig, ax = plt.subplots()
        ax.imshow(plot_neu_wordcloud(neu_keywords))
        ax.axis('off')
        st.pyplot(fig)

col1, col2 , col3 = st.columns(3)

with col1:
    st.markdown("<h4 style='text-align: center'>Opinions of the Top Highlight:</h4>", unsafe_allow_html=True)
    if pos_keywords:
        wc, top_word = plot_pos_wordcloud(pos_keywords, out = 'opinion')
        st.markdown(f"<h5 style='text-align: center'>{top_word}</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='nearest')
        ax.axis('off')
        st.pyplot(fig)

with col2:
    st.markdown("<h4 style='text-align: center'>Opinions of the Top Concern:</h4>", unsafe_allow_html=True)
    if neg_keywords:
        wc, top_word = plot_neg_wordcloud(neg_keywords, out='opinion')
        st.markdown(f"<h5 style='text-align: center'>{top_word}</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='nearest')
        ax.axis('off')
        st.pyplot(fig)

with col3:
    st.markdown("<h4 style='text-align: center'>Opinions of an Average Feature</h4>", unsafe_allow_html=True)
    if neu_keywords:
        wc, top_word = plot_neu_wordcloud(neu_keywords, out='opinion')
        st.markdown(f"<h5 style='text-align: center'>{top_word}</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='nearest')
        ax.axis('off')
        st.pyplot(fig)

