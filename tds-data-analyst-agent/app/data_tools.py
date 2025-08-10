import re
import io
import base64
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# --- URL utils ---

def extract_urls(text: str) -> list[str]:
    """
    Find and return all URLs from a given text.
    Uses a simple regular expression to match links.
    """
    return re.findall(r'https?://[^\s]+', text or "")


# --- Scraping ---

def _first_wikitable(soup: BeautifulSoup):
    """
    Find the first HTML table on the page with a 'wikitable' class.
    This is common for data tables on Wikipedia.
    """
    return soup.find("table", class_=lambda c: c and "wikitable" in c)

def scrape_wikipedia_table(url: str) -> pd.DataFrame | None:
    """
    Download a Wikipedia page and scrape the first 'wikitable' found.
    Returns a cleaned Pandas DataFrame, or None if no usable table.
    """
    try:
        # Request the page from Wikipedia
        r = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "tds-data-analyst-agent/1.0 (+edu)"},  # custom header
        )
        r.raise_for_status()  # throw error if status is not OK
    except Exception:
        return None  # return None if request fails

    # Parse HTML
    soup = BeautifulSoup(r.text, "html.parser")
    table = _first_wikitable(soup)
    if table is None:
        return None

    # Convert HTML table to DataFrame
    dfs = pd.read_html(str(table))
    if not dfs:
        return None

    df = dfs[0].copy()

    # Clean up column names
    df.columns = [str(c).strip() for c in df.columns]

    # Rename some common variations for consistency
    colmap = {}
    if "Film" in df.columns and "Title" not in df.columns:
        colmap["Film"] = "Title"
    for g in ["Worldwide gross", "Worldwide Gross", "Gross", "Worldwide box office"]:
        if g in df.columns:
            colmap[g] = "Worldwide gross"
            break
    df = df.rename(columns=colmap)

    # Convert 'Worldwide gross' to numeric values
    if "Worldwide gross" in df.columns:
        df["Worldwide gross"] = (
            df["Worldwide gross"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)  # remove non-numeric characters
        )
        df["Worldwide gross"] = pd.to_numeric(df["Worldwide gross"], errors="coerce")

    # Convert year, peak, and rank columns to numbers
    for col in ["Year", "Peak", "Rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --- Plotting ---

def plot_rank_vs_peak(df: pd.DataFrame) -> str:
    """
    Make a scatter plot of Rank vs Peak.
    Add a dotted red trend line.
    Return the plot as a base64 image string.
    """
    if not {"Rank", "Peak"}.issubset(df.columns):
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["Rank"], df["Peak"], label="Data points")

    # Try to fit a regression/trend line
    try:
        x = df["Rank"].dropna()
        y = df["Peak"].dropna()
        m, b = pd.Series(pd.np.polyfit(x, y, 1))  # slope (m) and intercept (b)
        xs = pd.Series([x.min(), x.max()])
        ax.plot(xs, m * xs + b, "r--", label="Trend")
    except Exception:
        pass  # If it fails, just skip the line

    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak")
    ax.legend()

    # Save plot to buffer and encode in base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


# --- Example request handler ---

def handle_question_request(req):
    """
    Example helper function for handling an incoming request.
    Reads questions.txt, scrapes data if Wikipedia link is found,
    and returns answers for 4 specific tasks.
    """
    # Read the text file from request
    qfile = req.files.get("questions.txt")
    question_text = (qfile.read().decode("utf-8", errors="ignore") if qfile else "") or ""

    # Extract any URLs
    urls = extract_urls(question_text)
    if not urls:
        return ["No URL found", "", 0.0, ""]

    # Scrape first Wikipedia URL found
    df = scrape_wikipedia_table(urls[0])
    if df is None or df.empty:
        return ["No table found", "", 0.0, ""]

    # Q1: Count movies > $2bn before 2000
    q1 = 0
    if {"Worldwide gross", "Year"}.issubset(df.columns):
        q1 = df[(df["Worldwide gross"] > 2_000_000_000) & (df["Year"] < 2000)].shape[0]

    # Q2: Earliest film over $1.5bn
    earliest_film = ""
    if {"Worldwide gross", "Year"}.issubset(df.columns) and "Title" in df.columns:
        high = df[df["Worldwide gross"] > 1_500_000_000].sort_values("Year")
        if not high.empty:
            earliest_film = str(high.iloc[0]["Title"])

    # Q3: Correlation between Rank and Peak
    corr = 0.0
    if {"Rank", "Peak"}.issubset(df.columns):
        corr_val = df["Rank"].corr(df["Peak"])
        corr = round(float(corr_val) if pd.notna(corr_val) else 0.0, 6)

    # Q4: Create plot
    img_uri = plot_rank_vs_peak(df)

    return [q1, earliest_film, corr, img_uri]
