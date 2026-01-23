import pandas as pd
import re
import spacy

# todo: What types of objects are most common in the archive?
# 1. DATA LOADING
# Load the Astronomy Picture of the Day (APOD) archive dataset
archive_data = pd.read_csv('project_files/apod.csv')
title_words_normalized = []

# 2. TEXT PRE-PROCESSING
# Iterate through every title to extract and clean individual words
for title in archive_data["title"]:
    found_words = re.findall(r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b', title) # Regex finds alphanumeric words and handles hyphenated words (e.g., "half-moon"). \b ensures we capture full words, not fragments

    title_words_normalized.extend([w.lower() for w in found_words]) # Normalize to lowercase to ensure "Space" and "space" are counted as the same word. this word_list is used later to join into a single string for spaCy processing

# 3. GLOSSARY PREPARATION
# Load a reference list of NASA astronomical terms: https://science.nasa.gov/universe/glossary
with open("project_files/nasa_glossary.txt", "r") as f:
    raw_glossary = f.read().splitlines()[:30] # only store the first 30 lines to keep the nested loop comparison (below) performant

# 4. NATURAL LANGUAGE PROCESSING (spaCy)
# Load the 'Large' model to ensure we have 'Word Vectors' for meaning-based comparison
nlp = spacy.load("en_core_web_lg")

# Convert our lists into spaCy Doc objects
blog_doc = nlp(" ".join(title_words_normalized)) # represents all words found in the APOD titles
glossary_doc = nlp(" ".join(raw_glossary)) # represents the technical terms we are looking for

found_matches = []

# 5. SEMANTIC SIMILARITY MATCHING
# Set 'run' to True to execute the heavy comparison logic
run = False

if run:
    # Nested Loop: Compare every word in the titles against every word in the glossary
    for title_token in blog_doc:
        for glossary_token in glossary_doc:
            # Only compare if both words have mathematical 'meaning vectors' available
            if title_token.has_vector and glossary_token.has_vector:
                similarity_score = title_token.similarity(glossary_token)

                # If the words are conceptually close (0.6+), save the word
                if similarity_score > 0.6:
                    found_matches.append(title_token.text)

    # Save the raw list of matched words to a temporary CSV
    match_series = pd.Series(found_matches)
    match_series.to_csv('similar_words.csv', index=False)

# 6. RESULTS ANALYSIS & EXPORT
# Read the matched words back in to calculate their frequencies
try:
    match_data = pd.read_csv("similar_words.csv")

    # Count occurrences of each word and convert from a Series back to a DataFrame
    final_counts = match_data.value_counts().reset_index()
    final_counts.columns = ['Word', 'Frequency']

    # Sort results: Highest frequency first; Alphabetical if frequencies are tied
    sorted_results = final_counts.sort_values(
        by=['Frequency', 'Word'],
        ascending=[False, True]
    )

    # Export the final report
    sorted_results.to_csv("final_word_counts.csv", index=False)
    print("Success: Final word counts saved to final_word_counts.csv")

except FileNotFoundError:
    print("Note: 'similar_words.csv' not found. Set 'run = True' to generate it.")

# TODO: Are any images posted more than once?
