import re
from typing import List, Dict
import collections
import argparse
import csv
import json 

def get_accuracy(result_list: List[Dict]):
    total_count = 0
    correct_count = 0
    for result in result_list:
        if result["pred"] == result["gt"]:
            correct_count += 1
        elif result["gt"] in result["pred"]:
            if len(result["pred"].split()) > 0 and result["pred"].split()[0] == result["gt"]:
                correct_count += 1
        total_count += 1
    return correct_count / total_count


def main(val_output_file_path):
    print(val_output_file_path)
    base_str_list = val_output_file_path.split("_")
    DATASET = base_str_list[2]
    LANG = base_str_list[3]

    if DATASET == 'shakespeare': # one line of poem is roughly 50 characters
        text = open('/scratch1/sghaneka/etok/tinyshake.txt', 'r').read() # don't worry we won't run out of file handles
    elif DATASET == 'custom':
        text = open(f'/home1/sghaneka/datasets_for_etok/{LANG}_10000.txt', 'r').read()
    else:
        raise ValueError(f"unknown dataset: {DATASET}")
    
    words = re.findall(r'\w+', text)
    frequency_groups = collections.defaultdict(list)
    for word, freq in collections.Counter(words).most_common():
        if freq > 45:
            frequency_groups["high_freq"].append(word)
        elif 45 > freq > 10:
            frequency_groups["mid_freq"].append(word)
        else:
            frequency_groups["low_freq"].append(word)
    
    reversed_frequency_groups = {tuple(v): k for k, v in frequency_groups.items()}

    eval_list = []
    try:
        with open(val_output_file_path, newline='') as csv_file:
            # csv_reader = csv.reader((line.replace('\0','') for line in csv_file), delimiter=',')
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                eval_dict = {}
                eval_dict["gt"] = row[-1]
                eval_dict["pred"] = row[-2]
                for key, val in reversed_frequency_groups.items():
                    if row[-1] in key:
                        eval_dict["frequency_group"] = val
                        break
                    else:
                        eval_dict["frequency_group"] = "undefined"
                eval_list.append(eval_dict)

        high_freq_list = [eval_dict for eval_dict in eval_list if eval_dict["frequency_group"] == "high_freq"]
        mid_freq_list = [eval_dict for eval_dict in eval_list if eval_dict["frequency_group"] == "mid_freq"]
        low_freq_list = [eval_dict for eval_dict in eval_list if eval_dict["frequency_group"] == "low_freq"]
        undefined_freq_list = [eval_dict for eval_dict in eval_list if eval_dict["frequency_group"] == "undefined"]

        eval_accuracy = {
            "high_freq_words":{"accuracy": get_accuracy(high_freq_list), "no_of_samples": len(high_freq_list)},
            "mid_freq_words":{"accuracy": get_accuracy(mid_freq_list), "no_of_samples": len(mid_freq_list)},
            "low_freq_words":{"accuracy": get_accuracy(low_freq_list), "no_of_samples": len(low_freq_list)},
            "undefined_freq_words":{"accuracy": get_accuracy(undefined_freq_list), "no_of_samples": len(undefined_freq_list)}
        }

        return eval_accuracy
    
    except IOError:
        return "Skipping file"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_output_file_paths", nargs='+', type=str)
    parser.add_argument("--output_file", type=str, default="eval_results.json")

    args = parser.parse_args()

    results = {val_output_file_path: main(val_output_file_path=val_output_file_path) for val_output_file_path in args.val_output_file_paths}
    
    with open(args.output_file, "w") as outfile:
        json.dump(results, outfile, indent = 4)