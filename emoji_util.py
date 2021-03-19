import requests
import pickle as pkl
from bs4 import BeautifulSoup
from datetime import datetime as dt

LINKS = ['http://www.unicode.org/emoji/charts/full-emoji-list.html', 
         'https://www.unicode.org/emoji/charts/full-emoji-modifiers.html']

def scrape_emojis(link):
    emojis = []
    unicode_vals = []
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    rows = soup.table.find_all('tr')
    for row in rows:
        try:
            cells = row.find_all('td')
            unicode_vals.append(cells[1].find('a').get('name'))
            emojis.append(cells[2].text)
        except Exception as e:
            pass
    return emojis, unicode_vals

def gather_emojis():
    all_emojis = []
    all_unicode = []
    for link in LINKS:
        emojis, unicode_vals = scrape_emojis(link)
        all_emojis += emojis
        all_unicode += unicode_vals
    return all_emojis, all_unicode

def load_emojis():
    all_emojis = pkl.load(open('emoji_symbols.p', 'rb'))
    all_unicode = pkl.load(open('emoji_unicode.p', 'rb'))
    all_unicode = list(map(lambda s: s.upper(), all_unicode))
    return all_emojis, all_unicode

def chunks(lst, n):
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    print('{} | Collecting emojis'.format(dt.now()))
    all_emojis, all_unicode = gather_emojis()
    pkl.dump(all_emojis, open('emoji_symbols.p', 'wb'))
    pkl.dump(all_unicode, open('emoji_unicode.p', 'wb'))
    print('{} | Complete'.format(dt.now()))