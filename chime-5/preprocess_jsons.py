import argparse
from .utils import get_files, to_decimal
from .align_parsing import build_alignments_hashtab, Alignment, get_utterance_word_boundaries
#from produce_lab import  build_alignments_hashtab, to_float, get_alignment_sad,
import os
import json
from copy import deepcopy


def write_new_json(jsons, aligments, outdir):

    count = 0
    for f in jsons:

        sess = f.split("/")[-1].split("_")[0].split(".json")[0]

        with open(f, "r") as json_file:
            data = json.load(json_file)
            new_json = deepcopy(data) # copy older json file

        for indx in range(len(data)):
            # iterate over all entries in the json

            if data[indx]["words"] == "[redacted]": # skip in the new json
                continue

            speaker = data[indx]["speaker"]
            original_time = to_decimal(data[indx]["start_time"]["original"])

            alignment_sad = Alignment([], [])

            try:
                alignment_sad = get_utterance_word_boundaries(original_time, aligments[sess + speaker])

            except EnvironmentError:
                count += 1
                print("Missing alignment: {} sess {} speaker {} time {}".format(count, sess, speaker, original_time))
                # continue


            new_json[indx]["alignments"] = alignment_sad.boundaries
            new_json[indx]["words"] = alignment_sad.words
            # correct for offsets
        with open(outdir + "/" + sess + ".json", 'w', encoding='utf-8') as f:
            json.dump(new_json, f,  ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Produce True lab file annotations for CHiME-5 from alignments.',
        usage='%(prog)s [options]')

    parser.add_argument("-j", "--json", type=str, help="jsons folder", metavar="STR", dest='json')
    parser.add_argument("-a", "--alignments", type=str, help="alignments folder", metavar="STR", dest='align')
    parser.add_argument("-o", "--out_path", type=str, help="output folder", metavar="STR", dest='out_path')

    args = parser.parse_args()

    jsons = get_files(args.json, "*.json", recursive=False)

    tot_alignments_files = get_files(args.align, "*.ctm", recursive=True)

    aligments = build_alignments_hashtab(tot_alignments_files)

    if not os.path.exists(args.out_path):
        # make save path if it does not exist
        os.makedirs(args.out_path)

    write_new_json(jsons, aligments, args.out_path)