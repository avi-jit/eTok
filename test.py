from tqdm import tqdm
import datasets
import re
from lingua import Language, LanguageDetectorBuilder

# lang = 'zh'
# ds = set()
# for i,row in tqdm(enumerate(datasets.load_dataset(f"squad_kor_v2", 
#      split="train", streaming=True, use_auth_token="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"))):
#     ds.add(row['context'])

# with open("/home1/sghaneka/datasets_for_etok/data/kor.txt", 'w') as out_file:
#     for line in tqdm(ds):
#         # out_file.write(line.replace("\n", " "))
#         out_file.write(line)
#         out_file.write("\n")

# languages = [Language.ENGLISH, Language.CHINESE]
# detector = LanguageDetectorBuilder.from_languages(*languages).build()
# clean_ds = [sentence[result.start_index:result.end_index] for sentence in tqdm(ds) for result in detector.detect_multiple_languages_of(sentence) if result.language.name == "CHINESE"]

# with open(f'clean_{lang}.txt', mode='wt', encoding='utf-8') as myfile:
#     for sentence in tqdm(clean_ds):  
#         clean_string_1 = re.sub("[A-Za-z]","",sentence)
#         myfile.write(re.sub(r'[-~].*', '', clean_string_1).replace("\n", "").strip())
#         myfile.write("\n")

# with open(f'clean_{lang}.txt', mode='wt', encoding='utf-8') as myfile:
#     for sentence in tqdm(ds):
#         for result in detector.detect_multiple_languages_of(sentence):
#             if result.language.name == "SPANISH":
#                 strings = sentence[result.start_index:result.end_index]
#                 clean_string_1 = re.sub("[A-Za-z]","",sentence)
#                 myfile.write(re.sub("\(\)", "", clean_string_1).replace("\n", "").strip())
#                 myfile.write("\n")

        # clean_string = sentence.replace("\n", "").strip()
        # myfile.write(clean_string)
        # myfile.write("\n")

with open("/home1/sghaneka/datasets_for_etok/ru_10000.txt", 'w') as out_file:
    with open("/home1/sghaneka/datasets_for_etok/data/ru.txt") as in_file:
        count = 0
        for line in in_file:
            if count == 10000:
                break
            out_file.write(line)
            count += 1