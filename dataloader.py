from datasets import load_dataset, concatenate_datasets
import pandas as pd

def load_polymath_dataset(language='en', split='all'):
    polymath = load_dataset('./datasets/PolyMath', language)
    if split == 'all':
        result = concatenate_datasets([polymath['top'], polymath['high'], polymath['medium'], polymath['low']])
    else:
        result = polymath[split]
    return result

def load_mgsm_dataset(language='en'):
    df = pd.read_csv(f'./datasets/mgsm/mgsm_{language}.tsv', sep='\t', header=None, names=['question', 'answer'])
    dataset = df.to_dict(orient='records')
    return dataset

def load_xquad_dataset(language='en'):
    xquad = load_dataset('./datasets/xquad', f"xquad.{language}")
    return xquad['validation']

def load_flores_dataset(language='cmn_Hans'):
    flores = load_dataset('./datasets/flores_plus', language)
    return flores['dev']