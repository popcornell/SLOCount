from ..utils.get_files import get_files
import re
import numpy as np
import soundfile as sf

class LibriWave:
    __slots__ = ["path", "id", "segments"]

    def __init__(self, path, id, segments):
        self.path = path
        self.id = id
        self.segments = segments

    def read(self, start, stop):

        audio, fs = sf.read(self.path, start=start, stop= stop )
        return audio

def parse_lab(sad_file, fs=16000):


    regex = re.compile("[0-9]*[.][0-9]+")

    segments = []

    with open(sad_file, "r") as f:
        for line in f:
            start = float(re.findall(regex, line)[0])
            stop = float(re.findall(regex, line)[1])

            start = int(start*fs)
            stop = int(np.ceil(stop*fs))

            segments.append([start, stop])


    return segments




def hash_librispeech(librispeech_traintest):

    #sha = sha3_512()
    hashtab = {}
    list_wav_id = []

    utterances = get_files(librispeech_traintest, "*.wav", recursive=True)

    for utt in utterances:

        id = utt.split("/")[-3]  # get folder name

        #sha.update(id.encode("utf-8"))

        id_hash = id #sha.digest()

        # get .lab segmentation

        lab_file = utt.split("-norm.wav")[0] + ".lab"

        speech = parse_lab(lab_file)

        if speech: # if no speech segments do not include file

            list_wav_id.append((LibriWave(utt, id_hash, speech)))

            if id_hash not in hashtab.keys():
                hashtab[id_hash] = [LibriWave(utt, id_hash, speech)]
            else:
                hashtab[id_hash].append(LibriWave(utt, id_hash, speech))


    return hashtab, list_wav_id