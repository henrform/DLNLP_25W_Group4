import json
import glob
from tqdm import tqdm
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

INPUT_DIR = "extracted_wiki"
OUTPUT_FILE = "wiki_for_bert.txt"


def main():
    input_files = glob.glob(f"{INPUT_DIR}/*/wiki_*")

    print(f"Processing {len(input_files)} files into {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for file_path in tqdm(input_files):
            with open(file_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line)
                        text = data['text']
                    except ValueError:
                        continue

                    if not text.strip():
                        continue

                    sentences = nltk.sent_tokenize(text)

                    for sent in sentences:
                        if sent.strip():
                            out_f.write(sent.strip() + "\n")

                    out_f.write("\n")


if __name__ == "__main__":
    main()