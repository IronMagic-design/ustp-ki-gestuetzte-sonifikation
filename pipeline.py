import csv
import os
import time
from typing import Optional
from dotenv import load_dotenv
import replicate
import requests

# Lädt Umgebungsvariablen aus der .env Datei
load_dotenv()

# Konfiguration der Modell-Versionen auf Replicate
LLAMA_MODEL = "meta/meta-llama-3-70b-instruct"
LLAVA_MODEL = "yorickvp/llava-13b:b5f621ed0425232b35000d238b108650573709ed607b3162870f032230e0108d"
AUDIOLDM_MODEL = "haoheliu/audioldm:b61392decdd6603261220b816182e6586b1a5d9431405d6030fd9d808928ad73"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
MAX_PROMPT_WORDS = 15  # Begrenzung zur Vermeidung von Repetitionen / Halluzinationen


# ==========================================
# 1. String-Sanitizing & Cut-off (Listing 3)
# ==========================================
def sanitize_prompt(raw_text: str, max_items: int = MAX_PROMPT_WORDS) -> str:
    """Bereinigt den Modell-Output und kappt repetitive Halluzinationen.

    Nimmt kommagetrennte Strings, filtert Duplikate und begrenzt die Länge.
    """
    if not raw_text:
        return ""

    # Zeilenumbrüche und Fülltext entfernen
    cleaned = raw_text.replace("\n", " ").strip()

    # Nach Kommas aufteilen und Leerzeichen trimmen
    items = [item.strip() for item in cleaned.split(",") if item.strip()]

    # Begrenzung auf die maximale Anzahl an akustischen Deskriptoren
    truncated_items = items[:max_items]

    return ", ".join(truncated_items)


# ==========================================
# 2. Retry-Handler gegen API-Timeouts (Listing 2)
# ==========================================
def call_replicate_with_retry(model_slug: str, input_params: dict, max_retries: int = MAX_RETRIES) -> Optional[any]:
    """Führt einen API-Aufruf mit Retry-Logik aus, um Cold Boots und Timeouts abzufangen."""
    for attempt in range(1, max_retries + 1):
        try:
            output = replicate.run(model_slug, input=input_params)
            return output
        except Exception as e:
            print(f"  [Warnung] Versuch {attempt}/{max_retries} fehlgeschlagen: {e}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
            else:
                print(f"  [Fehler] Modellaufruf endgültig fehlgeschlagen für {model_slug}.")
                return None


# ==========================================
# 3. KI-Konditionen (Listing 1 u. a.)
# ==========================================
def generate_condition_1_text(archive_text: str) -> str:
    """Kondition 1: Nur Text (Llama 3)."""
    system_instruction = (
        "You are a sound engineer. Analyze the following historical context description. "
        "Output ONLY a comma-separated list of English sound effects that match this physical environment. "
        "No introductory text, no conversational filler, no explanations. Just sound tags."
    )
    prompt = f"{system_instruction}\n\nHistorical Description:\n{archive_text}"

    output = call_replicate_with_retry(
        LLAMA_MODEL,
        {
            "prompt": prompt,
            "max_tokens": 80,
            "temperature": 0.2
        }
    )
    raw_prompt = "".join(output) if output else ""
    return sanitize_prompt(raw_prompt)


def generate_condition_2_image(image_url: str) -> str:
    """Kondition 2: Nur Bild (LLaVA-13b)."""
    prompt_instruction = (
        "Look at this historical photograph. Output ONLY a comma-separated list of English "
        "sound effects that fit the physical scene visible. No explanation, no sentences, just sounds."
    )

    output = call_replicate_with_retry(
        LLAVA_MODEL,
        {
            "image": image_url,
            "prompt": prompt_instruction,
            "max_tokens": 80,
            "temperature": 0.2
        }
    )
    raw_prompt = "".join(output) if output else ""
    return sanitize_prompt(raw_prompt)


def generate_condition_3_multimodal(image_url: str, archive_text: str) -> str:
    """Kondition 3: Multimodal - Bild + Text (LLaVA-13b)."""
    prompt_instruction = (
        "Look at this historical image and read the accompanying archive description. "
        "Output ONLY a comma-separated list of English sound effects that accurately "
        "represent the scene. No intro, no explanation, just sound descriptors."
    )
    full_prompt = f"{prompt_instruction}\n\nContext:\n{archive_text}"

    output = call_replicate_with_retry(
        LLAVA_MODEL,
        {
            "image": image_url,
            "prompt": full_prompt,
            "max_tokens": 100,
            "temperature": 0.2
        }
    )
    raw_prompt = "".join(output) if output else ""
    return sanitize_prompt(raw_prompt)


# ==========================================
# 4. Synthese via AudioLDM
# ==========================================
def synthesize_soundscape(prompt: str, output_path: str, duration: int = 10, guidance_scale: float = 2.5) -> bool:
    """Übergibt den Prompt an AudioLDM und speichert die resultierende WAV-Datei ab."""
    if not prompt:
        print(f"  [Übersprungen] Leerer Prompt für {output_path}.")
        return False

    print(f"  -> Synthetisiere Audio mit AudioLDM für Prompt: '{prompt}'")
    output_url = call_replicate_with_retry(
        AUDIOLDM_MODEL,
        {
            "text": prompt,
            "duration": str(duration),
            "guidance_scale": guidance_scale,
            "n_candidates": 1
        }
    )

    if output_url:
        # Replicate liefert eine URL zur erzeugten .wav Datei
        response = requests.get(output_url)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"  [Erfolg] Gespeichert unter: {output_path}")
            return True

    print(f"  [Fehler] Konnte Audio nicht herunterladen für {output_path}.")
    return False


# ==========================================
# 5. Batch-Processing der CSV-Pipeline
# ==========================================
def run_pipeline(csv_file_path: str, output_dir: str = "output_audio"):
    """Liest die Bild- und Metadatensätze ein und generiert die Audio-Spuren."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = row.get("id", "unknown")
            image_url = row.get("image_url")
            archive_text = row.get("text", "")

            print(f"\nVerarbeite Datensatz ID: {image_id}")

            # 1. Kondition: Nur Text
            prompt_text = generate_condition_1_text(archive_text)
            out_file_text = os.path.join(output_dir, f"{image_id}_text.wav")
            synthesize_soundscape(prompt_text, out_file_text)

            # 2. Kondition: Nur Bild
            prompt_image = generate_condition_2_image(image_url)
            out_file_image = os.path.join(output_dir, f"{image_id}_bild.wav")
            synthesize_soundscape(prompt_image, out_file_image)

            # 3. Kondition: Multimodal (Bild + Text)
            prompt_multi = generate_condition_3_multimodal(image_url, archive_text)
            out_file_multi = os.path.join(output_dir, f"{image_id}_multimodal.wav")
            synthesize_soundscape(prompt_multi, out_file_multi)


if __name__ == "__main__":
    # Pfad zu deinen historischen Ybbs-Daten
    CSV_PATH = "ybbs.csv"

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise ValueError("REPLICATE_API_TOKEN ist nicht gesetzt. Bitte in .env eintragen.")

    run_pipeline(CSV_PATH)
