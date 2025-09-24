# save as douban_to_excel.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from tqdm import tqdm  # progress bar (pip install tqdm)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
}


def parse_movie_node(node):
    """Extract data from a single movie 'info' node (BeautifulSoup object)."""
    # title (sometimes multiple titles: the first .title is main)
    title_tag = node.find('span', class_='title')
    title = title_tag.text.strip() if title_tag else ''

    # link to movie page
    # the parent structure usually: <div class="hd"><a href="...">...<span class="title">...</span></a></div>
    link = ''
    # the hd block sits beside info
    hd = node.find_previous_sibling('div', class_='hd')
    if hd:
        a = hd.find('a')
        if a and a.get('href'):
            link = a['href']

    # info text (the descriptive paragraph under .bd > p)
    info_p = node.find('div', class_='bd').p if node.find(
        'div', class_='bd') else None
    info_text = info_p.get_text(separator=' ', strip=True) if info_p else ''

    # rating
    rating_tag = node.find('span', class_='rating_num')
    rating = rating_tag.text.strip() if rating_tag else 'N/A'

    # quote (may be missing)
    quote_tag = node.find('p', class_='quote')
    quote = quote_tag.span.text.strip() if (quote_tag and quote_tag.span) else ''

    # number of critics (in the same div as rating_num, in a span)
    critics = node.find('span', property='v:best').find_next_sibling('span').text.replace('人评价','').strip() if node.find('span') else 'N/A'

    return {
        'title': title,
        'info_text': info_text,
        'rating': rating,
        'quote': quote,
        'critics': critics,
        'link': link
    }


def scrape_top250(output_excel='douban_top250.xlsx'):
    rows = []
    # Douban Top250 pages: start=0,25,50,...,225
    starts = list(range(0, 226, 25))
    for i in tqdm(starts, desc="pages"):
        url = f"https://movie.douban.com/top250?start={i}&filter="
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            # ensure correct encoding
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'html.parser')

            # find all movie blocks; original code used div.info — keep that
            nodes = soup.find_all('div', class_='info')
            for node in nodes:
                try:
                    data = parse_movie_node(node)
                    data['page_start'] = i
                    rows.append(data)
                except Exception as e:
                    # skip single node on parse error but keep scraping
                    print(f"parse error on page start={i}: {e}")

        except Exception as e:
            print(f"request error for start={i}: {e}")

        # polite random delay
        time.sleep(random.uniform(1, 3))

    # build dataframe and save to excel
    df = pd.DataFrame(rows, columns=[
                      'page_start', 'title', 'info_text', 'rating', 'quote', 'critics'])
    # optional: strip extra whitespace
    for col in ['title', 'info_text', 'quote', 'critics']:
        df[col] = df[col].astype(str).str.strip()

    # save
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"saved {len(df)} rows to {output_excel}")


if __name__ == "__main__":
    scrape_top250()
