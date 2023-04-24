import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
def sentiment(review, sentiment_pipeline, tokenizer, max_length=512):
    # Calculate the adjusted max_length considering the special tokens
    adjusted_max_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)

    # Tokenize the input text and truncate it to the adjusted maximum length
    tokens = tokenizer.encode(review, truncation=True, max_length=adjusted_max_length)

    # Convert the truncated tokens back into a string
    truncated_review = tokenizer.decode(tokens)

    # Perform sentiment analysis on the truncated review
    return sentiment_pipeline(truncated_review)[0]



model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
URL = 'http://insideairbnb.com/get-the-data/'
OUTPUT_DIR = 'WebData'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MAX_RETRIES = 5
RETRY_DELAY = 5  # Delay between retries in seconds

for attempt in range(MAX_RETRIES):
    try:
        u = urlopen(URL)
        html = u.read().decode('utf-8')
        u.close()
        break
    except URLError as e:
        if attempt < MAX_RETRIES - 1:
            print(f"Error: {e}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            print(f"Failed to connect after {MAX_RETRIES} attempts. Exiting.")
            exit(1)

soup = BeautifulSoup(html, "html.parser")

# Start of the for loop
for link in soup.select('a[href^="http://"]'):
    href = link.get('href')
    if not any(href.endswith(x) for x in ['.csv','.xls','.xlsx','.gz','.geojson']):
        continue

    # Check if 'amsterdam' is in the URL
    if 'amsterdam' not in href.lower():
        continue

    filename = href.rsplit('/', 1)[-1]
    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        #href = href.replace('http://','https://')
        print("Downloading %s to %s..." % (href, filepath))
        urlretrieve(href, filepath)
        print("Done.")
    else:
        print(f"{filename} already exists. Skipping download.")
# End of the for loop
for filename in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.isfile(file_path):
        print(f'Processing file: {file_path}')
        if  'calendar.csv.gz' in file_path:
            pdcalender = pd.read_csv(file_path, compression='gzip')
        if 'listings.csv.gz' in file_path:
            pdlistings = pd.read_csv(file_path, compression='gzip')
        if 'reviews.csv.gz' in file_path:
            pdreviews = pd.read_csv(file_path, compression='gzip')
print('***************************************************************')
# Wrap the DataFrame iterator with tqdm for displaying the progress bar
if 'sentiment' not in pdreviews.columns:
    tqdm.pandas(desc="Processing comments")
    pdreviews['sentiment']=pdreviews['comments'].progress_apply(lambda x: sentiment(x, sentiment_pipeline, tokenizer))
    pdreviews.to_csv(os.path.join(OUTPUT_DIR,'reviews.csv.gz'),compression='gzip')
print(pdreviews.head())
print('***************************************************************')
