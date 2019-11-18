import re

def build_alignments_hashtab(alignments_files):
    '''

    :param alignments_files: list of force aligner output files
    :return: dictionary with keys sess+ speaker with the list of alignment
    '''
    hash_align = {}

    for f in alignments_files:

        alignment = CHiME5Alignment(f)
        speaker = alignment.get_device()
        sess = alignment.get_session()

        if sess + speaker not in hash_align.keys():

            hash_align[sess + speaker] = [alignment]

        else:

            hash_align[sess + speaker].append(alignment)

    return hash_align

class Alignment:
    '''
    General word-level alignment class, boundaries is a list of [start, stop] for each word. words is a list of words.
    '''

    def __init__(self, boundaries, words):
        assert len(boundaries) == len(words) # check that coundaries == n_words otherwise something should be wrong with
                                            #  force aligner
        self.boundaries = boundaries
        self.words = words


    def __len__(self):
        return len(self.words)

class CHiME5Alignment:
    '''
    CHiME5 forced aligner output file class
    '''

    def __init__(self, file):

        self.file = file

    def get_session(self):

        return self.file.split("/")[-1].split("_")[1]

    def get_device(self):

        return self.file.split("/")[-1].split("_")[0]

    def get_start(self):

        return self.file.split("/")[-1].split("-")[1]

    def get_stop(self):

        return self.file.split("/")[-1].split("-")[-1].split(".ctm")[0]

    def get_boundaries(self):  # parse the .ctm forced aligner output to get the speaker activity

        boundaries = []

        regex_start = re.compile("-[0-9]*")
        regex_offset = re.compile("[0-9]*[.][0-9]+")

        # include also words

        with open(self.file, "r") as f:

            words = []

            for line in f:

                word = line.strip(" \n").split(" ")[-1] # ignore \n
                if word == "<eps>" or word == "[noise]": # are there other segments which should be not included ?
                    continue  # skip silence and noise

                match = re.findall(regex_start, line)  # ask if this can be done only at beginning
                start, stop = match[0], match[1]  # ignore words
                start, stop = int(start[1:]) * 0.01, int(stop[1:]) * 0.01

                start_word, offset_word = re.findall(regex_offset, line)

                start_word, offset_word = float(start_word), float(offset_word)

                boundaries.append([start + start_word, start + start_word + offset_word])

                if start + start_word + offset_word > stop:
                    print("Alignment out of the original boundary. You may want to check this!")

                words.append(word)

        return Alignment(boundaries, words)


def get_utterance_word_boundaries(utterance_time, alignments):
    # alignments are supposed to be ordered by their starting time

    for al in alignments: # find the correct alignment file
        #TODO not very efficient can be made much more efficient
        if int(al.get_start()) - 1 <= int(utterance_time * 100) <= int(al.get_start()) + 1:
            # open alignment file
            return al.get_boundaries()

    # print("Missing utterance")
    raise EnvironmentError
