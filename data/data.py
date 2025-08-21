import requests
import dateparser
import json
import os
import time

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime


MALE_NEWS_URL_IT = "https://www.trentinovolley.it/it/news/superlega"
MALE_NEWS_URL_EN = "https://www.trentinovolley.it/en/news/superlega"
FEMALE_NEWS_URL_IT = "https://www.trentinovolley.it/it/news/femminile"
YOUTH_NEWS_URL_IT = "https://www.trentinovolley.it/it/news/giovanile-news"


MALE_MATCHES_URL = "https://www.trentinovolley.it/it/superlegaa/tutte-le-date" # "https://www.trentinovolley.it/en/superlega/save-the-dates"
FEMALE_MATCHES_URL = "https://www.trentinovolley.it/it/femminile/a1femminile-tutte-le-date"


# GET DATA FROM DATABASE =============================================

def get_price(intent: str, slot_values: dict) -> float:
    """
    Retrieves the price for a specific intent and its associated slot values.
    """
    with open("data/prices.json", "r") as f:
        prices = json.load(f)

        if intent == "buy_tickets":
            team = slot_values.get("team", "male")
            sector = slot_values.get("sector", "grandstand")
            seasonal = slot_values.get("seasonal", False)
            reduced = slot_values.get("reduced_price", False)
            number_of_tickets = slot_values.get("number_of_tickets", 1)
            price = prices[intent][team][sector]
            if seasonal: price *= prices[intent]["multipliers"].get("seasonal", 10)
            if reduced: price *= prices[intent]["multipliers"].get("reduced", 0.8)
            price *= int(number_of_tickets)
            return str(price)+ "€"

        elif intent == "buy_merchandise":
            item = slot_values.get("item", "shirt")
            price = prices[intent].get(item, 0.0)
            price *= int(slot_values.get("quantity", 1))
            return str(price) + "€"

        return None


def standardize_date(date_str: str) -> str:
    """
    Standardizes the date format. The input date can be in many formats:
    - 19/10/2025
    - 19th October 2025
    - 19 ottobre 2025
    The output is in the format YYYY-MM-DD.
    """
    dt = dateparser.parse(date_str, languages=['en', 'it'])  
    if dt is None: return None
    return dt.strftime("%Y-%m-%d")


def standardize_time(time_str: str) -> str:
    """
    Standardizes the time format. The input time can be in many formats:
    - 14:30
    - 2:30 PM
    - 14:30:00
    The output is in the format HH:MM.
    """
    dt = dateparser.parse(time_str, languages=['en', 'it'])
    if dt is None: return None
    return dt.strftime("%H:%M")


# GET DATA FROM WEBSITE =============================================

def get_articles(team="male", lang="en", num_articles=3) -> list[dict]:
    """
    Fetches the latest articles for the specified team and language.
    """
    # Set url
    if team == "male":
        url = MALE_NEWS_URL_EN if lang == "en" else MALE_NEWS_URL_IT
    elif team == "female":
        url = FEMALE_NEWS_URL_IT
        if lang == "en":
            print("\033[93mWarning: English articles not be available for female team.\033[0m")
            lang = "it"
    elif team == "youth":
        url = YOUTH_NEWS_URL_IT
        if lang == "en":
            print("\033[93mWarning: English articles not be available for youth teams.\033[0m")
            lang = "it"

    # Check if dataset exists
    articles_dataset_path = f"data/articles_{team}_{lang}.json"
    if os.path.exists(articles_dataset_path):
        with open(articles_dataset_path, "r") as f:
            articles_dataset = json.load(f)
    else:
        articles_dataset = []

    # If dataset last modified is within 2 hours, use cached data
    if articles_dataset and (time.time() - os.path.getmtime(articles_dataset_path) < 7200):
        return articles_dataset[:num_articles]  # Return only the most recent articles
    
    new_articles = []

    try:
        resp = requests.get(url)
        resp.raise_for_status()  # will raise an error for bad responses
        soup = BeautifulSoup(resp.text, 'html.parser')
        container = soup.find('div', class_='lista-articoli-news pag-news')
        if container:
            for a_tag in container.find_all('a', href=True):
                date_div = a_tag.find('div', class_='data')
                title_div = a_tag.find('div', class_='titolo')
                date = date_div.get_text(strip=True) if date_div else None
                title = title_div.get_text(strip=True) if title_div else None
                link = urljoin(url, a_tag['href'])
                if date and title:
                    if not any(article['link'] == link for article in articles_dataset):
                        new_articles.append({
                            "date": standardize_date(date),
                            "title": title,
                            "link": link
                        })
    except requests.RequestException as e:
        print(f"Error fetching articles: {e}")

    articles_dataset.extend(new_articles)
    articles_dataset.sort(key=lambda x: x['date'], reverse=True) # Most recent first

    # Save to dataset
    with open(articles_dataset_path, "w") as f:
        json.dump(articles_dataset, f, indent=4)

    return articles_dataset[:num_articles]  # Return only the most recent articles


def get_matches(team="male", num_matches=3) -> list[dict]:
    """
    Fetch the upcoming matches for the specified team.
    """
    # Set url
    if team == "male":
        url = MALE_MATCHES_URL
    else:
        url = FEMALE_MATCHES_URL

    # Check if dataset exists
    matches_dataset_path = f"data/matches_{team}.json"
    if os.path.exists(matches_dataset_path):
        with open(matches_dataset_path, "r") as f:
            matches_dataset = json.load(f)
    else:
        matches_dataset = []

    # If dataset last modified is within 2 hours, use cached data
    if matches_dataset and (time.time() - os.path.getmtime(matches_dataset_path) < 7200):
        return matches_dataset[:num_matches]  # Return only the upcoming matches

    matches = []

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        matches_div = soup.find('div', id='tutteLeDate')
        if matches_div:
            table = matches_div.find('table')
            if table:
                rows = table.find_all('tr')
                # Skip the header row
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) != 3:
                        continue
                    date_text = cols[0].get_text(strip=True)
                    time_text = cols[1].get_text(strip=True)
                    match_text = cols[2].get_text(separator=" ", strip=True)
                    matches.append({
                        "date": standardize_date(date_text),
                        "time": standardize_time(time_text),
                        "match": match_text
                    })
    except requests.RequestException as e:
        print(f"Error fetching matches: {e}")
    
    # Filter and sort matches
    matches = [match for match in matches if match['date'] >= datetime.now().isoformat()]
    matches.sort(key=lambda x: (x['date']))

    # Save to dataset
    with open(matches_dataset_path, "w") as f:
        json.dump(matches, f, indent=4)

    return matches[:num_matches]  # Return only the upcoming matches


def get_results(team="male", num_results=3) -> list[dict]:
    """
    Retrieve the match results for the specified team.
    """

    results_dataset_path = f"data/results_{team}.json"
    with open(results_dataset_path, "r") as f:
        results_dataset = json.load(f)
    return results_dataset[:num_results]  # Return only the most recent results


# CHECK FIELDS VALIDITY =============================================

def get_user_settings(user_id: str) -> dict:
    with open("data/user_settings.json", "r") as f:
        user_settingss = json.load(f)
    for user_settings in user_settingss:
        if user_settings["user_id"] == user_id:
            return user_settings
    return {}


def update_user_settings(user_settings_data: dict):
    """
    Update the user_settings.json with the given user_settings data.
    """
    user_id = user_settings_data.get("user_id")
    with open("data/user_settings.json", "r") as f:
        user_settingss = json.load(f)
    for i, user_settings in enumerate(user_settingss):
        if user_settings["user_id"] == user_id:
            user_settingss[i] = user_settings_data
            break
    with open("data/user_settings.json", "w") as f:
        json.dump(user_settingss, f, indent=4)


def is_valid_integer(value, frame):
    """Check if the value is a valid integer, lower than 10."""
    try:
        return 0 < int(value) < 10
    except (ValueError, TypeError):
        return False

def is_valid_date(value, frame):
    """Check if the value is a valid date."""
    try:
        date = standardize_date(value)
        datetime.strptime(date, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False

def is_valid_match_date(value, frame):
    """Check if the value is a valid date, and there are matches scheduled (Note: "season" is a special case)"""
    if isinstance(value, str) and value.lower().startswith("season"): # Special case for season tickets
        return True
    try:
        team = frame.get("slots", {}).get("team", "male").lower()
        date = standardize_date(value)
        # Check if there are matches scheduled for this date
        upcoming_matches = get_matches(team)
        upcoming_matches = [match for match in upcoming_matches if match['date'] == date]
        return len(upcoming_matches) > 0
    except (ValueError, TypeError):
        return False

def is_valid_address(value, frame):
    """Check if the address is valid (non-empty string)."""
    return isinstance(value, str) and len(value.strip()) > 0

DEFAULT_SIZE = "unique"
def is_valid_size(value, frame):
    """Check if the value is a valid size.
    - For shirt: S, M, L, XL
    - For other items: unique
    """
    item = frame.get("slots", {}).get("item", "").lower()
    if item and item.lower() != "shirt":
        return value == DEFAULT_SIZE
    return value in ["s", "m", "l", "xl"]

def is_valid_specific_field(value, frame, return_list=False):
    """A valid specific field is a slot that exists in buy_tickets or buy_merchandise."""
    if return_list:
        valid_list = list(SLOTS["buy_tickets"].keys()) + list(SLOTS["buy_merchandise"].keys()) + ["any"]
        return set(valid_list)
    return value in SLOTS["buy_tickets"] or value in SLOTS["buy_merchandise"] or value in ["any"]


SLOTS = {
    "buy_tickets": {
        "team": "male or female",
        "sector": "vip, best, grandstand, or curve",
        "season_ticket": "boolean (if the user wants a season ticket)",
        "reduced_price": "boolean (if for an under 25 or over 65)",
        "number_of_tickets": "integer",
        "date": "the date of the match in the format YYYY-MM-DD (for season tickets, the value is 'season')"
    },
    "buy_merchandise": {
        "team": "male or female",
        "item": "shirt, scarf, hat, or ball",
        "size": "S, M, L, XL for shirts; unique for other items",
        "quantity": "integer",
        "delivery_address": "the delivery address, in Italy"
    },
    "matches_schedule": {
        "team": "male or female"
    },
    "get_news": {
        "team": "male, female, or youth"
    },
    "get_results": {
        "team": "male or female"
    },
    "information_request": {
        "topic": "tickets, merchandise",
        "specific_field": "the specific field the user is asking about, or 'any' if not specified"
    }
}

SLOTS_ACCEPTED_VALUES = {
    "buy_tickets": {
        "team": ["male", "female"],
        "sector": ["vip", "best", "grandstand", "curve"],
        "season_ticket": [True, False],
        "reduced_price": [True, False],
        "number_of_tickets": is_valid_integer,
        "date": is_valid_match_date
    },
    "buy_merchandise": {
        "team": ["male", "female"],
        "item": ["shirt", "scarf", "hat", "ball"],
        "size": is_valid_size,
        "quantity": is_valid_integer,
        "delivery_address": is_valid_address
    },
    "matches_schedule": {
        "team": ["male", "female"]
    },
    "get_news": {
        "team": ["male", "female", "youth"]
    },
    "get_results": {
        "team": ["male", "female"]
    },
    "information_request": {
        "topic": ["tickets", "merchandise"],
        "specific_field": is_valid_specific_field
    }
}

def is_slot_value_accepted(intent, slot, value, frame):
    """Check if the slot value is accepted for the given intent. (Note: works for every slot except match date)"""
    if isinstance(value, str):
        value = value.lower()
    accepted = SLOTS_ACCEPTED_VALUES.get(intent, {}).get(slot)
    if accepted is None:
        return False
    if callable(accepted):
        return accepted(value, frame)
    return value in accepted


# TODO: Make these real partial orders (i.e., some intents (actions) can be at same level)
INTENTS_PARTIAL_ORDER = [
    "information_request",
    "matches_schedule",
    "get_news",
    "get_results",
    "buy_tickets",
    "buy_merchandise",
    "out_of_domain"
]

ACTIONS_PARTIAL_ORDER = [
    "request_info",
    "confirmation",
    "cancel",
    "fallback_policy"
]

