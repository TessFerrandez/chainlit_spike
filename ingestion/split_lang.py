from pathlib import Path
import spacy_fastlang
import spacy
from langdetect import detect
from tqdm import tqdm


def extract_language_langdetect(text: str) -> str:
    detected_lang = detect(text)
    return detected_lang


def copy_file(file_path: Path, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(file_path.read_text())


def process_file(file_path: Path):
    text = file_path.read_text()
    ld_lang = extract_language_langdetect(text)
    out_path = Path("./data/parsed") / ld_lang / file_path.parents[-2] / file_path.name

    copy_file(file_path, out_path)


if __name__ == "__main__":
    for file_path in tqdm(Path("./data/hr_kb").glob("**/*.html")):
        process_file(file_path)
