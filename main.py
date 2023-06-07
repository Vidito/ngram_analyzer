import streamlit as st
import pandas as pd
from collections import Counter
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
import string

st.set_page_config(
        page_title="N-Gram Analyzer",
)
nltk.download('punkt')

# Function to extract n-grams from a text
def extract_ngrams(text, n):
    tokens = word_tokenize(text)
    ngrams_list = ngrams(tokens, n)

    # Filter out unwanted characters
    unwanted_chars = string.punctuation
    filtered_ngrams = [ngram for ngram in ngrams_list if all(token not in unwanted_chars for token in ngram)]

    return list(filtered_ngrams)

# Function to process the uploaded files
def process_files(files, n, num_common):
    all_ngrams = []

    for file in files:
        try:
            text = file.read().decode("utf-8")
            file_ngrams = extract_ngrams(text, n)
            all_ngrams.extend(file_ngrams)
        except UnicodeDecodeError:
            st.error('oops! Make sure you have the correct file format')
    # Count the frequency of each n-gram
    frequency_counter = Counter(all_ngrams)

    # Get the most common n-grams
    most_common = frequency_counter.most_common(num_common)

    # Create a DataFrame from the most common n-grams
    df = pd.DataFrame(most_common, columns=['N-gram', 'Frequency'])

    return df

# Streamlit app
def main():
    st.title("N-gram Frequency Analyzer")
    st.success("Upload your text files and get the frequencies of n-grams.")

    # Upload files
    uploaded_files = st.file_uploader("Upload Text Files", accept_multiple_files=True)
    if uploaded_files:
        st.write("Processing files...")

        # let's create some columns
        col1, col2 = st.columns(2)
        with col1:
            # Choose n-gram size
            st.info('N-Gram Size')
            n = st.slider(" ", min_value=3, max_value=5, value=4)
        with col2:
            # Specify number of most common n-grams
            st.info('Number of most common n-grams to display')
            num_common = st.number_input(" ", min_value=1, value=10)

        # Process the uploaded files
        df = process_files(uploaded_files, n, num_common)
        st.subheader(f"Most Common {n}-grams:")
        st.dataframe(df)
        # Save the DataFrame to a CSV file
        # Download button
        csv = df.to_csv(index=False)
        button_label = "Download CSV"
        st.download_button(
            label=button_label,
            data=csv,
            file_name="ngram_frequency.csv",
            mime="text/csv"
        )



    else:
        st.info("Please upload text files.")

if __name__ == "__main__":
    main()
