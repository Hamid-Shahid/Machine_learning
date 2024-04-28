import json
import os
import re
from bs4 import BeautifulSoup
import pandas as pd
import requests

search_keywords = ['fashion','climate']
headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0',
}

def validity_condition(dic):
    return (not dic.get('isLocked', True)) and dic.get('reading_time', 0) >= 4

def get_apollo_satte(html_text):
    start_index = html_text.find("window.__APOLLO_STATE__ = ")
    if start_index == -1:
        return False

    end_index = html_text.find("</script>", start_index)
    if end_index == -1:
        return False

    json_string = html_text[start_index + len("window.__APOLLO_STATE__ = "):end_index]

    try:
        apollo_state_dict = json.loads(json_string)
        return apollo_state_dict
    except json.JSONDecodeError as e:
        return False
    
def search_keyword(keyword):
    results = []
    results_file = f"{keyword}_results.csv"
    url = 'https://medium.com/_/graphql'
    graph_query = open('searchquery.txt', 'r').read()
    results_limit = 50
    page = 0
    while page <= 10:
        page += 1
        json_data = [
            {
                'operationName': 'SearchQuery',
                'variables': {
                    'query': keyword,
                    'pagingOptions': {
                        'limit': results_limit,
                        'page': page,
                    },
                    'withUsers': False,
                    'withTags': False,
                    'withPosts': True,
                    'withCollections': False,
                    'withLists': False,
                    'peopleSearchOptions': {
                        'filters': 'highQualityUser:true OR writtenByHighQulityUser:true',
                        'numericFilters': 'peopleType!=2',
                        'clickAnalytics': True,
                        'analyticsTags': [
                            'web-main-content',
                        ],
                    },
                    'postsSearchOptions': {
                        'filters': 'writtenByHighQualityUser:true',
                        'clickAnalytics': True,
                        'analyticsTags': [
                            'web-main-content',
                        ],
                    },
                    'publicationsSearchOptions': {
                        'clickAnalytics': True,
                        'analyticsTags': [
                            'web-main-content',
                        ],
                    },
                    'tagsSearchOptions': {
                        'numericFilters': 'postCount>=1',
                        'clickAnalytics': True,
                        'analyticsTags': [
                            'web-main-content',
                        ],
                    },
                    'listsSearchOptions': {
                        'clickAnalytics': True,
                        'analyticsTags': [
                            'web-main-content',
                        ],
                    },
                    'searchInCollection': False,
                    'collectionDomainOrSlug': 'medium.com',
                },
                'query': graph_query,
            },
        ]

        response = requests.post(url=url, headers=headers, json=json_data)
        if response.ok:
            json_data = response.json()
            results += get_attrs(json_data)
            pd.DataFrame(results).to_csv(results_file, index=False)

            count_valid_dics = sum(1 for d in results if validity_condition(d))
            if count_valid_dics >= 15:
                break

    valid_dics = [d for d in results if validity_condition(d)]
    return valid_dics

def get_attrs(json_data):
    extracted_data = []
    for item in json_data[0]["data"]["search"]["posts"]["items"]:
        extracted_data.append({
                    "id": item["id"],
                    "title": item["title"],
                    "author": item["creator"]["name"],
                    "follower_count": item["creator"]["socialStats"]["followerCount"],
                    "medium_url": item["mediumUrl"],
                    "clap_count": item["clapCount"],
                    "published_at": item["firstPublishedAt"],
                    "image_url": item["previewImage"]["id"] if "previewImage" in item else None,
                    "reading_time": item["readingTime"],
                    "uniqueSlug": item["uniqueSlug"],
                    "isLocked": item["isLocked"]
                })
    return extracted_data

for keyword in search_keywords:
    folder_path = f'{keyword} data'
    count = 0
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dics = search_keyword(keyword)
    print(list(map(lambda x: x['medium_url'], dics)))
    # results = pd.read_csv(f'{keyword}_results.csv').to_dict(orient='records')
    # dics = [d for d in results if validity_condition(d)]

    for dic in dics:
        try:
            response = requests.get(dic['medium_url'], headers=headers)
            
            if response.ok:
                apollo_state = get_apollo_satte(response.text)
                if apollo_state:
                    count += 1
                    text = ''

                    for key, value in apollo_state.items():
                        if key.startswith("Paragraph"):
                            text += ' ' + value.get('text', '')

                    file_path = os.path.join(folder_path, f'{count}.txt')
                    f = open(file_path, 'w', encoding='utf-8')
                    f.write(text)
                    f.flush()
                    f.close()
        except:
            pass
