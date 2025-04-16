import os
import requests

from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from pprint import pprint

load_dotenv()

def scrape_linkedin_profile(url: str, mock: bool = True) -> str:
    ''' 
    Scrape LinkedIn profile data from a given URL.
    Parameters:
    - url (str): The LinkedIn profile URL to scrape.
    - mock (bool): If True, use mock data instead of scraping. Default is False.
    Returns:
    - str: The scraped profile data in JSON format.
    '''
    if mock:
        linkedin_url = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"
        response = requests.get(
            url=linkedin_url, 
            timeout=5
        )
    else:
        api_ep = "https://api.scrapin.io/enrichment/profile/linkedin"
        params = {
            "api_key": os.environ["SCRAPIN_API_KEY"],
            "linkedIn_Url": url,
        }
        response = requests.get(
            url=api_ep,
            params=params,
            timeout=5
        )
    # print(response.josn())
    data = response.json().get("person")
    data = {
        k:v
        for k, v in data.items() 
        if v not in ([], "", "", None) and k not in ["certifications"]
    }
    # pprint(data)
    return data

if __name__ == "__main__":
    url = "https://www.linkedin.com/in/eden-marco/"
    data = scrape_linkedin_profile(url, mock=True)
    print(data)