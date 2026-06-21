def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_sms_spam():
    import io
    import zipfile
    import urllib.request

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            with zf.open('SMSSpamCollection') as f:
                df = pd.read_csv(
                    f, sep='\t', header=None, names=['label', 'text'], encoding='latin-1'
                )
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(clean_text)
    return df


def load_ag_news(subset=AG_NEWS_SUBSET):
    """Load AG News from HuggingFace JSONL (no datasets library required)."""
    import json
    import urllib.request

    base = 'https://huggingface.co/datasets/sh0416/ag_news/resolve/main'
    rows = []
    for split in ('train', 'test'):
        with urllib.request.urlopen(f'{base}/{split}.jsonl') as resp:
            for line in resp:
                item = json.loads(line)
                rows.append({
                    'label': item['label'] - 1,
                    'text': clean_text(f"{item['title']} {item['description']}"),
                })
    df = pd.DataFrame(rows)

    if subset and subset < len(df):
        per_class = subset // df['label'].nunique()
        df = (
            df.groupby('label', group_keys=False)
            .sample(n=per_class, random_state=RANDOM_SEED)
            .reset_index(drop=True)
        )
    return df


sms_df = load_sms_spam()
ag_df = load_ag_news()

print('SMS Spam:', sms_df.shape, sms_df['label'].value_counts().to_dict())
print('AG News:', ag_df.shape, ag_df['label'].value_counts().to_dict())

dataset_summary = pd.DataFrame([
    {'Dataset': 'SMS Spam', 'Classes': 2, 'Total': len(sms_df),
     'Train': int(len(sms_df)*(1-TEST_SIZE)), 'Test': int(len(sms_df)*TEST_SIZE)},
    {'Dataset': 'AG News', 'Classes': 4, 'Total': len(ag_df),
     'Train': int(len(ag_df)*(1-TEST_SIZE)), 'Test': int(len(ag_df)*TEST_SIZE)},
])
dataset_summary