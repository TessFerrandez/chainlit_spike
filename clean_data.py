# prepare by
# adding your files to data/hr_kb
# create a folder called data/clean/hr_kb/en
# with two folders purpose, and qa
import os
import spacy
import spacy_fastlang
from langchain_community.document_loaders import UnstructuredHTMLLoader


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")

file_directory = 'data/hr_kb'

file_number = 0
for filename in os.listdir(file_directory):
    f = os.path.join(file_directory, filename)

    if os.path.isfile(f):
        with open(f, 'r') as file:
            try:
                text = file.read()
                doc = nlp(text)
                language = doc._.language
            except Exception as e:
                print(f"ERROR: {e} on file {filename}")

            if language == 'en':
                loader = UnstructuredHTMLLoader(f)
                data = loader.load()
                first_row = data[0].page_content.splitlines()[0]

                if first_row == 'Purpose':
                    path = f"data/clean/hr_kb/en/purpose/{filename}"
                elif first_row.startswith("Q:"):
                    path = f"data/clean/hr_kb/en/qa/{filename}"
                else:
                    path = None

                if path:
                    with open(path, "w") as dest:
                        dest.write(text)
    file_number += 1
    print(file_number)
