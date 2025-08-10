import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Import our own modules
from app import analysis, scraping
from app.data_tools import extract_urls

# Create a FastAPI app instance
app = FastAPI(title="Data Analyst Agent API")

@app.post("/api/")
async def analyze_data(
    questions: UploadFile = File(...),  # required: questions.txt
    files: list[UploadFile] | None = File(default=None)  # optional: CSV files
):
    """
    Main API endpoint to handle data analysis requests.
    It:
      1. Reads the questions from a text file.
      2. Loads any provided CSV files into Pandas DataFrames.
      3. Finds Wikipedia links in the questions.
      4. Scrapes the first Wikipedia table if available.
      5. Chooses the right DataFrame (scraped or uploaded).
      6. Answers the questions using the analysis module.
    """

    # --- Step 1: Read the question text from the uploaded file ---
    question_text = await questions.read()
    question_str = question_text.decode("utf-8", errors="ignore").strip()

    # --- Step 2: Load uploaded CSVs into a dictionary {filename: DataFrame} ---
    dataframes: dict[str, pd.DataFrame] = {}
    if files:
        for f in files:
            try:
                if f.filename.lower().endswith(".csv"):
                    content = await f.read()
                    df = pd.read_csv(io.BytesIO(content))
                    dataframes[f.filename] = df
            except Exception:
                # If one file fails, skip it instead of crashing
                continue

    # --- Step 3: Look for URLs inside the question text ---
    urls = extract_urls(question_str)

    # --- Step 4: If any Wikipedia URL is found, scrape the first table ---
    wiki_df = None
    if urls:
        for url in urls:
            if "wikipedia.org" in url:
                wiki_df = await scraping.scrape_wikipedia_table(url)
                if wiki_df is not None:
                    break  # stop after the first successful scrape

    # --- Step 5: Decide which DataFrame to use for answering ---
    # Priority: Wikipedia table > First uploaded CSV > None
    df_to_use = wiki_df if wiki_df is not None else (next(iter(dataframes.values()), None))

    # --- Step 6: Use our analysis module to answer the questions ---
    answers = await analysis.answer_questions(question_str, df_to_use)

    # Return answers as JSON
    return JSONResponse(content=answers)


# --- Run locally if file is executed directly ---
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # default port 8000
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
