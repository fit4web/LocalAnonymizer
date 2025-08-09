# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi import Form
from fastapi.requests import Request
import html
from typing import Tuple, Dict
from pydantic import BaseModel
from typing import Dict, Tuple
import spacy
import csv
import uvicorn
from pathlib import Path
import requests
import time
import os
import re
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)


# ----------------- Konfiguration -----------------
API_KEY = ""
API_BASE_URL = "https://api.openai.com/v1/chat/completions"
API_MODEL = "gpt-4o-mini"

ENABLE_CACHE = True   # Caching ein-/ausschalten
CACHE_TTL_SECONDS = 300  # 5 Minuten

# ----------------- Request/Response-Modelle -----------------
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    anonymized_text: str
    mapping: Dict[str, str]
    llm_response: str
    final_output: str

# ----------------- Einfaches In-Memory-Cache -----------------
_cache_store = {}

def cache_get(key: str):
    if not ENABLE_CACHE:
        return None
    entry = _cache_store.get(key)
    if entry:
        value, timestamp = entry
        if time.time() - timestamp <= CACHE_TTL_SECONDS:
            return value
        else:
            _cache_store.pop(key, None)
    return None

def log_error(request_text: str, error_message: str):
    now = datetime.now()
    year_folder = Path(__file__).parent / "log" / str(now.year)
    month_folder = year_folder / f"{now.month:02d}"
    month_folder.mkdir(parents=True, exist_ok=True)

    logfile = month_folder / "errors.log"
    log_entry = (f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] "
                 f"Fehler: {error_message}\n"
                 f"Request-Text: {request_text}\n"
                 f"{'-'*60}\n")

    # 🔹 Logfile schreiben
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # 🔹 Farbig ins Terminal ausgeben
    print(
        f"{Fore.YELLOW}[{now.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{Fore.RED}Fehler: {error_message}\n"
        f"{Fore.WHITE}Request-Text: {request_text}\n"
        f"{Fore.LIGHTBLACK_EX}{'-'*60}{Style.RESET_ALL}",
        flush=True
    )

def cache_set(key: str, value):
    if ENABLE_CACHE:
        _cache_store[key] = (value, time.time())

def highlight_pii_in_text(original_text: str, mapping: Dict[str, str]) -> str:
    """
    Hebt die anonymisierten PII-Elemente im Originaltext fett und rot hervor.
    """
    escaped_text = html.escape(original_text)
    # Sortiere längere Einträge zuerst, damit keine Überlappungen kaputt gehen
    sorted_pii = sorted(mapping.values(), key=len, reverse=True)
    for pii in sorted_pii:
        escaped_pii = html.escape(pii)
        # Ersetze alle Vorkommen mit markiertem HTML
        escaped_text = escaped_text.replace(
            escaped_pii,
            f'<b style="color:red;">{escaped_pii}</b>'
        )
    # Zeilenumbrüche zu <br> für HTML
    return escaped_text.replace("\n", "<br>")

# ----------------- Anonymisierer -----------------
class Anonymizer:
    def __init__(self, csv_path: str):
        try:
            self.nlp = spacy.load("de_core_news_lg")
        except OSError:
            raise RuntimeError("Bitte installiere das Modell: python -m spacy download de_core_news_lg")

        self.regexes = {
            "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "Geburtsdatum": r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',
            "ADRESSE": r'\b[A-Za-zßäöüÄÖÜ\s\-]+\d{1,3}(?:\s?[a-zA-Z])?\s+\d{5}\s+[A-Za-zßäöüÄÖÜ\s\-]+\b'
        }

        self.pii_data = {}
        self.csv_loaded = False
        try:
            self.pii_data = self._load_csv(csv_path)
            self.csv_loaded = True
        except FileNotFoundError as e:
            err_msg = f"Warnung: CSV-Datei nicht gefunden: {csv_path}. Fallback ohne CSV aktiviert."
            print(err_msg, flush=True)
            # log_error("", err_msg)  # Optional Logging

    def _load_csv(self, csv_path: str) -> Dict[str, list]:
        pii_dict = {}
        with open(csv_path, newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    value, category = row[0].strip(), row[1].strip()
                    pii_dict.setdefault(category, []).append(value)
        return pii_dict

    def _generate_placeholder(self, category: str, counters: Dict[str, int]) -> str:
        key = category
        counters[key] = counters.get(key, 0) + 1
        return f"[{key}_{counters[key]}]"

    def anonymize(self, text: str) -> Tuple[str, Dict[str, str]]:
        mapping = {}
        placeholder_counters = {}
        all_ents = []

        # --- Schritt 1: Alle potenziellen Entitäten aus allen Quellen sammeln ---

        # Priorität 1: Regex-basierte Entitäten (am spezifischsten)
        for category, pattern in self.regexes.items():
            for match in re.finditer(pattern, text):
                value = match.group(0)
                if category == "Geburtsdatum":
                    try:
                        date_obj = datetime.strptime(value, "%d.%m.%Y")
                        if (datetime.now() - date_obj).days < 365:
                            continue
                    except ValueError:
                        continue

                all_ents.append({
                    "start": match.start(), "end": match.end(), "value": value,
                    "category": category, "priority": 1
                })

        # Priorität 2: CSV-basierte Entitäten (benutzerdefiniert)
        if self.csv_loaded:
            for category, values in self.pii_data.items():
                # Lange Werte zuerst, um Teilübereinstimmungen zu vermeiden (z.B. "Meier" vor "Meierling")
                for value in sorted(values, key=len, reverse=True):
                    start = 0
                    while True:
                        idx = text.find(value, start)
                        if idx == -1:
                            break
                        # Vornamen/Nachnamen bekommen eine höhere Priorität als andere CSV-Einträge
                        priority = 2 if category in ["VORNAME", "NACHNAME"] else 3
                        all_ents.append({
                            "start": idx, "end": idx + len(value), "value": value,
                            "category": category, "priority": priority
                        })
                        start = idx + len(value)

        # Priorität 4: SpaCy-basierte Entitäten (am allgemeinsten)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("PER", "LOC", "ORG"):
                all_ents.append({
                    "start": ent.start_char, "end": ent.end_char, "value": ent.text,
                    "category": ent.label_, "priority": 4
                })

        # --- Schritt 2: Überlappungen basierend auf Priorität auflösen ---

        # Sortiere nach Startposition, dann Priorität, dann Länge (längere zuerst)
        sorted_ents = sorted(all_ents, key=lambda x: (x['start'], x['priority'], -(x['end'] - x['start'])))

        unique_ents = []
        if sorted_ents:
            # Iteriere durch die sortierten Entitäten und verwerfe die mit niedrigerer Priorität bei Überlappung
            last_ent = None
            for ent in sorted_ents:
                # Wenn es keine letzte Entität gibt oder keine Überlappung, füge sie hinzu
                if last_ent is None or ent['start'] >= last_ent['end']:
                    if last_ent is not None:
                        unique_ents.append(last_ent)
                    last_ent = ent
                # Bei Überlappung gewinnt die mit der höheren Priorität (bereits durch Sortierung sichergestellt)
                # Wir tun also nichts und verwerfen die aktuelle `ent`

            if last_ent is not None:
                unique_ents.append(last_ent)

        # --- Schritt 3: Text mit den finalen Entitäten ersetzen ---

        # Logik zur Wiederverwendung von Platzhaltern für identische PII-Werte
        value_to_placeholder = {}

        # Absteigend nach Startposition sortieren, um den Text von hinten nach vorne zu ersetzen
        final_ents = sorted(unique_ents, key=lambda x: x['start'], reverse=True)

        for ent in final_ents:
            value = ent['value']
            category = ent['category']

            if value in value_to_placeholder:
                placeholder = value_to_placeholder[value]
            else:
                placeholder = self._generate_placeholder(category, placeholder_counters)
                mapping[placeholder] = value
                value_to_placeholder[value] = placeholder

            text = text[:ent['start']] + placeholder + text[ent['end']:]

        return text, mapping
    
# ----------------- Deanonymisierung -----------------
def deanonymize(text: str, mapping: Dict[str, str]) -> str:
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text

# ----------------- OpenAI-kompatibler API-Call -----------------
def send_to_llm(anonymized_text: str) -> str:
    cache_key = f"llm:{anonymized_text}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
            {"role": "user", "content": anonymized_text}
        ]
    }

    try:
        resp = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        llm_output = data["choices"][0]["message"]["content"]

        cache_set(cache_key, llm_output)
        return llm_output
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim LLM-Request: {e}")

# ----------------- FastAPI App -----------------
app = FastAPI(title="PII Anonymization Service")

CSV_PATH = Path(__file__).parent / "patients.csv"
anonymizer = Anonymizer(str(CSV_PATH))

@app.get("/test", response_class=HTMLResponse)
async def test_form():
    return """
    <html>
    <head>
      <title>Text Anonymizer</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 40px;
          background-color: #f9f9f9;
        }
        h1 {
          color: #333;
          margin-bottom: 20px;
        }
        textarea {
          width: 100%;
          font-family: monospace;
          font-size: 16px;
          padding: 10px;
          border: 1px solid #ccc;
          border-radius: 4px;
          resize: vertical;
          box-sizing: border-box;
        }
        button {
          margin-top: 10px;
          padding: 10px 20px;
          font-size: 16px;
          background-color: #007BFF;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        button:hover {
          background-color: #0056b3;
        }
        table {
          border-collapse: collapse;
          width: 100%;
          margin-top: 30px;
          background: white;
          box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        th, td {
          border: 1px solid #ddd;
          padding: 15px;
          vertical-align: top;
          font-family: monospace;
          font-size: 15px;
          word-wrap: break-word;
        }
        th {
          background-color: #f2f2f2;
          text-align: left;
        }
        /* PII Hervorhebung links */
        .pii-highlight {
          color: red;
          font-weight: bold;
        }
        a {
          display: inline-block;
          margin-top: 20px;
          text-decoration: none;
          color: #007BFF;
          font-weight: bold;
        }
        a:hover {
          text-decoration: underline;
        }
      </style>
    </head>
    <body>
      <h1>Text Anonymizer</h1>
      <form method="post" action="/test">
        <textarea name="text" rows="15" placeholder="Füge hier deinen Text mit Personendaten ein..."></textarea><br>
        <button type="submit">Senden</button>
      </form>
    </body>
    </html>
    """

@app.post("/test", response_class=HTMLResponse)
async def test_form_post(text: str = Form(...)):
    print("Raw input text:", repr(text))
    anonymized_text, mapping = anonymizer.anonymize(text)
    original_highlighted = highlight_pii_in_text(text, mapping)
    anonymized_escaped = html.escape(anonymized_text).replace("\n", "<br>")
    html_content = f"""
    <html>
    <head><title>Text Anonymizer Ergebnis</title></head>
    <body>
      <h1>Text Anonymizer Ergebnis</h1>
      <table>
          <tr>
              <th style="width:50%; background:#fdd;">Originaltext (PII rot/fett)</th>
              <th style="width:50%; background:#dfd;">Anonymisierter Text</th>
          </tr>
          <tr>
              <td style="vertical-align: top;">{original_highlighted}</td>
              <td style="vertical-align: top; font-family: monospace;">{anonymized_escaped}</td>
          </tr>
      </table>
      <a href="/test">Zurück</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ----------------- API-Endpunkt -----------------
@app.post("/process_text")
async def process_text(req: TextRequest):
    try:
        anonymized_text, mapping = anonymizer.anonymize(req.text)
        # DEV_MODE: wenn kein API_KEY gesetzt, gebe HTML zurück
        if not API_KEY:
            log_error(req.text, "Kein API-Key gesetzt - > im DEV_MODE gestartet")
            original_highlighted = highlight_pii_in_text(req.text, mapping)
            anonymized_escaped = html.escape(anonymized_text).replace("\n", "<br>")
            html_content = f"""
            <html>
            <head><title>Dev Mode Anonymizer Check</title></head>
            <body>
                <h2>Dev Mode: Anonymisierungsprüfung</h2>
                <table border="1" cellpadding="10" style="border-collapse: collapse; width: 100%;">
                    <tr>
                        <th style="width:50%; background:#fdd;">Originaltext (PII rot/fett)</th>
                        <th style="width:50%; background:#dfd;">Anonymisierter Text</th>
                    </tr>
                    <tr>
                        <td style="vertical-align: top; font-family: monospace;">{original_highlighted}</td>
                        <td style="vertical-align: top; font-family: monospace;">{anonymized_escaped}</td>
                    </tr>
                </table>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content, status_code=200)

        # Sonst normale Verarbeitung mit API-Call
        llm_response = send_to_llm(anonymized_text, req.text)
        final_output = deanonymize(llm_response, mapping)
        return TextResponse(
            anonymized_text=anonymized_text,
            mapping=mapping,
            llm_response=llm_response,
            final_output=final_output
        )

    except Exception as e:
        error_msg = str(e)
        log_error(req.text, error_msg)
        # Im DEV_MODE auch Fehler als HTML ausgeben
        if not API_KEY:
            html_content = f"""
            <html><body><h2 style="color:red;">Fehler bei der Verarbeitung</h2>
            <pre>{html.escape(error_msg)}</pre>
            </body></html>
            """
            return HTMLResponse(content=html_content, status_code=200)

        return TextResponse(
            anonymized_text=req.text,
            mapping={},
            llm_response="",
            final_output=f"Fehler bei der Verarbeitung: {error_msg}"
        )
    
@app.get("/health/csv_status")
def csv_status():
    """
    Gibt zurück, ob die patients.csv aktuell geladen und aktiv ist.
    Nützlich für Monitoring oder Healthchecks.
    """
    return {"csv_loaded": anonymizer.csv_loaded}
    


# ----------------- Starter -----------------
if __name__ == "__main__":
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Erika", "VORNAME"])
            writer.writerow(["Mustermann", "NACHNAME"])
            writer.writerow(["erika@mail.de", "EMAIL"])
            writer.writerow(["0123456789", "TELEFON"])
            writer.writerow(["Testweg 1", "ADRESSE"])
    if not API_KEY:
        log_error("DEV-MODE gestartet", "Kein API-Key (vollständig) konfiguriert")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    