import spacy
import re

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def parse_query(text):
    doc = nlp(text.lower())

    # Extract age
    age_match = re.search(r"(\d{1,3})[- ]?year[- ]?old", text.lower())
    if not age_match:
        age_match = re.search(r"(\d{1,3})[mMfF]\b", text.lower())  # Handles "46M" or "46F"
    age = int(age_match.group(1)) if age_match else None

    # ✅ FIXED: Extract policy duration correctly (removed escape error)
    duration_match = re.search(r"(\d{1,2})[- ]?(month[s]?|year[s]?)", text.lower())
    duration = f"{duration_match.group(1)} {duration_match.group(2)}" if duration_match else None

    # Extract gender
    gender = None
    if "male" in text.lower() or re.search(r"\b\d{1,3}m\b", text.lower()):
        gender = "male"
    elif "female" in text.lower() or re.search(r"\b\d{1,3}f\b", text.lower()):
        gender = "female"

    # Extract city by checking for known cities (expandable list)
    cities = ["pune", "mumbai", "delhi", "bangalore"]
    city = next((c for c in cities if c in text.lower()), None)

    # Extract procedure from known list (can be extended)
    procedures = ["knee surgery", "heart surgery", "fracture", "hospitalization"]
    procedure = next((p for p in procedures if p in text.lower()), None)

    return {
        "age": age,
        "gender": gender,
        "procedure": procedure,
        "location": city,
        "policy_duration": duration
    }

if __name__ == "__main__":
    query = input("Enter your query: ")
    result = parse_query(query)
    print(result)
