import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from scipy.stats import linregress
import re
import asyncio
from typing import Optional

# Helper function to clean and standardize the DataFrame
def _normalize_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    # If no data or empty, just return None
    if df is None or df.empty:
        return None

    out = df.copy()
    # Clean column names (remove spaces, etc.)
    out.columns = [str(c).strip() for c in out.columns]

    # Find the "Gross" column, even if it's named differently
    gross_aliases = ["Gross", "Worldwide gross", "Worldwide Gross", "Worldwide box office"]
    gross_col = next((c for c in gross_aliases if c in out.columns), None)
    if gross_col and "Gross" not in out.columns:
        out = out.rename(columns={gross_col: "Gross"})

    # Convert "Gross" column values into billions (GrossNum)
    if "Gross" in out.columns:
        def parse_gross(x):
            s = str(x)
            # Keep only numbers, dots, commas
            s = re.sub(r"[^\d.,]", "", s)
            s = s.replace(",", "")
            try:
                val = float(s)
            except:
                return np.nan
            return val / 1e9  # convert dollars to billions
        out["GrossNum"] = out["Gross"].apply(parse_gross)

    # Convert some columns to numbers if they exist
    for c in ("Year", "Rank", "Peak"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# Main function to answer questions based on the DataFrame
async def answer_questions(question_str: str, df: pd.DataFrame):
    """
    Go through each question and try to answer using the given dataframe.
    Returns a list of answers in the same order as questions.
    """

    # Try to split the text into separate questions like "1. something"
    questions = [line.strip() for line in question_str.splitlines()
                 if re.match(r'^\d+\.\s*', line.strip())]
    # If no numbering, treat the whole text as one question
    if not questions and question_str.strip():
        questions = [question_str.strip()]

    answers = []
    ndf = _normalize_df(df)  # Cleaned DataFrame

    for q in questions:
        q_lower = q.lower()

        # Question type 1: How many $2 bn movies before 2000?
        if ("how many" in q_lower) and ("$2 bn" in q_lower) and ("before 2000" in q_lower):
            if ndf is None or not {"GrossNum", "Year"}.issubset(ndf.columns):
                answers.append(0)
            else:
                count = ndf[(ndf["GrossNum"] >= 2.0) & (ndf["Year"] < 2000)].shape[0]
                answers.append(int(count))

        # Question type 2: Earliest film that made over $1.5 bn
        elif ("earliest film" in q_lower) and ("1.5 bn" in q_lower):
            if ndf is None or not {"GrossNum", "Year"}.issubset(ndf.columns) or "Title" not in ndf.columns:
                answers.append("")
            else:
                filtered = ndf[ndf["GrossNum"] > 1.5].dropna(subset=["Year"])
                if filtered.empty:
                    answers.append("")
                else:
                    earliest = filtered.sort_values("Year", kind="mergesort").iloc[0]
                    answers.append(str(earliest["Title"]))

        # Question type 3: Correlation between Rank and Peak
        elif ("correlation" in q_lower) and ("rank" in q_lower) and ("peak" in q_lower):
            if ndf is None or not {"Rank", "Peak"}.issubset(ndf.columns):
                answers.append(0.0)
            else:
                sub = ndf[["Rank", "Peak"]].dropna()
                if sub.empty:
                    answers.append(0.0)
                else:
                    corr = sub["Rank"].corr(sub["Peak"])
                    answers.append(round(float(corr) if pd.notna(corr) else 0.0, 6))

        # Question type 4: Scatterplot of Rank vs Peak with regression line
        elif ("scatterplot" in q_lower) and ("rank" in q_lower) and ("peak" in q_lower):
            if ndf is None or not {"Rank", "Peak"}.issubset(ndf.columns):
                answers.append("")
            else:
                sub = ndf[["Rank", "Peak"]].dropna()
                if sub.empty:
                    answers.append("")
                else:
                    # Create scatter plot
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(sub["Rank"], sub["Peak"], label="Data points")

                    # Add regression line
                    slope, intercept, *_ = linregress(sub["Rank"], sub["Peak"])
                    x_vals = np.array([sub["Rank"].min(), sub["Rank"].max()])
                    y_vals = intercept + slope * x_vals
                    ax.plot(x_vals, y_vals, "r--", label="Regression line")

                    ax.set_xlabel("Rank")
                    ax.set_ylabel("Peak")
                    ax.legend()
                    plt.tight_layout()

                    # Save plot as base64 string
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    data_uri = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

                    # Cut down very large strings
                    if len(data_uri) > 100_000:
                        data_uri = data_uri[:100_000]
                    answers.append(data_uri)

        # If question doesn't match any rule
        else:
            answers.append("Question not recognized or data missing")

    return answers
