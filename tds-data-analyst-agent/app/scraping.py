import httpx
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional

async def scrape_wikipedia_table(url: str) -> Optional[pd.DataFrame]:
    """
    Try to fetch the first 'wikitable' from a Wikipedia page.
    Steps:
      1. Make an async HTTP request to the page.
      2. Look for the first table with class 'wikitable'.
      3. Convert the HTML table into a Pandas DataFrame.
      4. Clean up column names and rename common variants.
    Returns:
      A Pandas DataFrame if found, otherwise None.
    """

    # --- Step 1: Download the page using an async HTTP client ---
    try:
        async with httpx.AsyncClient(
            timeout=20,  # stop waiting after 20 seconds
            headers={"User-Agent": "tds-data-analyst-agent/1.0 (+edu)"}  # custom header so Wikipedia doesn't block us
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()  # raise error if HTTP status not 200
    except Exception:
        # If request fails (bad URL, timeout, etc.), return None
        return None

    # --- Step 2: Parse HTML to find the first wikitable ---
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_=lambda c: c and "wikitable" in c)
    if table is None:
        return None  # No table found on the page

    # --- Step 3: Convert HTML table to a Pandas DataFrame ---
    try:
        df_list = pd.read_html(str(table))  # pandas parses the HTML table
    except ValueError:
        return None  # Parsing failed
    if not df_list:
        return None  # No valid tables found

    # Take the first table found
    df = df_list[0].copy()

    # --- Step 4: Clean up column names (remove extra spaces) ---
    df.columns = [str(c).strip() for c in df.columns]

    # --- Step 5: Rename known column variations for consistency ---
    if "Film" in df.columns and "Title" not in df.columns:
        df = df.rename(columns={"Film": "Title"})

    return df
