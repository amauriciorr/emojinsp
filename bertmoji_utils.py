from bertmoji_model import * 

MODEL = "cardiffnlp/twitter-roberta-base"
LABEL_MAPPING = {1.0:'Y', 0.0:'N'}

def load_model(checkpoint_path):
    '''
    helper function for loading model, training + validation performance
    and tokenizer in one-go
    '''
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    train_performance = checkpoint['train_performance']
    valid_performance = checkpoint['valid_performance']
    model = AutoModel.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # necessary for correctly loading checkpoint
    tokenizer.add_tokens(["[USER]"])
    model.resize_token_embeddings(len(tokenizer)) 

    bertmoji_classifier = bertmoji(model)
    bertmoji_classifier.load_state_dict(checkpoint['model'])
    return bertmoji_classifier, tokenizer, train_performance, valid_performance

def tokenize_data(data, tokenizer, return_df=False):
    '''
    function to tokenize tweets and emoji sentence, also creates attention mask.
    data is returned as a TensorDataset.
    '''
    if isinstance(data, pd.DataFrame):
        df_ = data.copy()
    else:
        df_ = pd.DataFrame(data, columns = ['tweet', 'emoji_sentence'])
    tweets = df_['tweet'] + ' ' + tokenizer.sep_token + ' ' + df_['emoji_sentence']
    max_sentence_length = int(tweets.str.len().max())
    tokenized_tweets = []
    attn_masks = []
    for idx, tweet in tweets.items():
        encoded = tokenizer.encode_plus(tweet, padding='max_length',
                                        truncation=True, max_length=max_sentence_length)
        tokenized_tweets.append(encoded['input_ids'])
        attn_masks.append(encoded['attention_mask'])
    tokenized_tweets = torch.tensor(tokenized_tweets, dtype=torch.long)
    attn_masks = torch.tensor(attn_masks, dtype=torch.long)
    if return_df:
        return TensorDataset(tokenized_tweets, attn_masks), df_
    else:
        return TensorDataset(tokenized_tweets, attn_masks)

def run_batch(model, data, tokenizer, device, human_readable=False):
    '''
    wrap tokenize_data in a single test-execution, return results in a dataframe
    for analysis
    '''
    dataset, df = tokenize_data(data, tokenizer, return_df=True)
    loader = DataLoader(dataset, batch_size=1)
    model.eval()
    model.to(device)
    predictions = []
    for step, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks = batch
        logits = model(input_ids, attention_masks)
        logits = logits.to(device)
        preds = torch.round(torch.sigmoid(logits))
        predictions.append(preds.item())
    df['prediction'] = predictions
    if human_readable:
        df['prediction'] = df['prediction'].map(LABEL_MAPPING)
        if 'label' in df.columns:
            df['label'] = df['label'].map(LABEL_MAPPING)
    return df
