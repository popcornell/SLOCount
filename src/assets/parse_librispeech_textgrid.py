import tgt
from ..utils.get_files import get_files
from ..utils.segmentation import smooth_segments
from .librispeech import LibriWave


def read_phone_alignment(read_textgrid):
    read_tier_phones = read_textgrid.get_tier_by_name('phones')
    n_phones = len(read_tier_phones)
    phones = []
    start_time = []
    end_time = []
    for i_w in range(1, n_phones):
        phones.append(read_tier_phones[i_w].text)
        start_time.append(read_tier_phones[i_w].start_time)
        end_time.append(read_tier_phones[i_w].end_time)

    return phones, start_time, end_time

# thanks to Maurizio Omologo
def read_word_alignment(read_textgrid):
    read_tier_words = read_textgrid.get_tier_by_name('words')
    n_words = len(read_tier_words)
    words = []
    start_time = []
    end_time = []
    for i_w in range(n_words):
        words.append(read_tier_words[i_w].text)
        start_time.append(read_tier_words[i_w].start_time)
        end_time.append(read_tier_words[i_w].end_time)

    return words, start_time, end_time


def get_textgrid_sa(mfa_file):

    read_textgrid = tgt.read_textgrid(mfa_file)
    [words, start_time, end_time] = read_word_alignment(read_textgrid)
    assert len(words) == len(start_time) == len(end_time)
    stack = []
    for i in range(len(words)):

        if words[i] == "":
            continue

        if stack:
            if start_time[i] > stack[-1][-1]:
                stack.append([start_time[i], end_time[i]])
                #print("non contiguos word")
            else:
                stack[-1][-1] = end_time[i]

        else:
            stack.append([start_time[i], end_time[i]])

    return stack


def build_hashtable_textgrid(textgrid_dir):
    hashtab = {}

    mfa_files = get_files(textgrid_dir, "*.TextGrid", recursive=True)

    for f in mfa_files:

        # get .lab segmentation

        filename = f.split("/")[-1].split(".TextGrid")[0]

        if filename not in hashtab.keys():
            hashtab[filename] = f
        else:
            raise EnvironmentError

    return hashtab

def build_example_list(librispeech_dir, hp):


    hashtab = {}
    list_wav_id = []

    hashgrid = build_hashtable_textgrid(hp.data.textgrid_dir[0])

    utterances = get_files(librispeech_dir, "*.wav", recursive=True)

    for utt in utterances:

        id = utt.split("/")[-3]  # get folder name

        # sha.update(id.encode("utf-8"))

        id_hash = id  # sha.digest()

        # get .lab segmentation

        filename = utt.split("/")[-1].split("-norm.wav")[0]
        if filename not in hashgrid.keys():
            print("Missing Alignment file: {}".format(utt))
            continue
        speech = get_textgrid_sa(hashgrid[filename])
        if hp.data.smooth:
            speech = smooth_segments(speech, *hp.data.smooth)


        if speech:  # if no speech segments do not include file

            speech = [[int(s*hp.data.fs),int(e*hp.data.fs)] for s,e in speech]

            list_wav_id.append((LibriWave(utt, id_hash, speech)))

            if id_hash not in hashtab.keys():
                hashtab[id_hash] = [LibriWave(utt, id_hash, speech)]
            else:
                hashtab[id_hash].append(LibriWave(utt, id_hash, speech))

    return hashtab, list_wav_id







