"""
This module provides functionality for multilingual support.
"""

import deepl
import os
from dotenv import load_dotenv


class Translator:
    """A class to handle translation using the DeepL API."""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv('DEEPL_API_KEY')
        if not api_key:
            raise ValueError("DEEPL_API_KEY not found in environment variables.")
        self.translator = deepl.Translator(api_key)

    def translate(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            target_lang = "EN-GB"
        if target_lang == "it":
            target_lang = "IT"
        result = self.translator.translate_text(text, target_lang=target_lang)
        return result.text

    def __call__(self, text: str, target_lang: str) -> str:
        """Translate text to the specified target language."""
        return self.translate(text, target_lang)
