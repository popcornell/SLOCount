import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import numpy as np
#from .utils.hashchime import hashchime, hashjson
from tqdm import tqdm
import pickle
import json
from .utils.segmentation import smooth_segments, count_overlaps, merge_contiguos
from .assets.chime5 import hashchime
from .utils.time import to_decimal
import librosa
import random
from .utils.audio import peakNorm


def create_dataloader(hp, args, train):
    if train:
        return DataLoader(dataset=CHiME5Dataset(hp, args, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          )
    else:
        return DataLoader(dataset=CHiME5Dataset(hp, args, False),
                          batch_size=hp.train.batch_size, shuffle=False, num_workers=hp.train.num_workers)


class Example:
    __slots__ = ["array", "boundaries", "segments"]

    def __init__(self, array_obj, boundaries, segments):
        self.array = array_obj
        self.boundaries = boundaries
        self.segments = segments

    def read_example(self):

        return self.array.read_data(self.boundaries), self.segments

    def get_segments(self):

        return self.segments


class SynthExample:
    __slots__ = ["examples"]

    def __init__(self, examples):
        self.examples = examples

    def read_example(self):

        audio, segments = self.examples[0].read_example()
        target_len = len(audio)
        gain = random.randint(-10, 3)
        audio = peakNorm(audio, gain)

        total_segments = segments

        for sp in range(1, len(self.examples)):
            # note discard files without segments before
            c_segments = self.examples[sp].get_segments()
            c_segment = random.choice(c_segments)  # sample a random segment from utterance
            start, stop = self.examples[sp].boundaries
            self.examples[sp].boundaries = c_segment[0] + start, start + c_segment[1]

            c_speech, _ = self.examples[sp].read_example()
            self.examples[sp].boundaries = start, stop

            true_start = 0
            true_end = c_segment[1] - c_segment[0]

            gain = random.randint(-10, 3)
            c_speech = peakNorm(c_speech, gain)

            len_speech = len(c_speech)

            # this can be shorter or longer than hp.data.audio_len
            # if it is longer take a portion
            # choose a random offset

            start_bound = -(true_end - 8000) if true_end - 8000 > 0 else 0
            end_bound = target_len - 1 - 8000 - (
                true_start)  # at least always include half second and true start
            rand_offset = random.randint(start_bound, end_bound)

            if rand_offset < 0:
                end = min(target_len, len_speech - abs(rand_offset))
                audio[0:end, :] += c_speech[abs(rand_offset):min(target_len, len_speech) + abs(rand_offset),:]

                # regarding segmentation
                c_start = true_start - abs(rand_offset) if abs(rand_offset) < true_start else 0
                c_end = true_end - abs(rand_offset) if target_len > true_end - abs(rand_offset) else end
            else:

                end = min(target_len, rand_offset + len_speech)
                audio[rand_offset:end, :] += c_speech[0:min(len_speech, target_len - rand_offset)]

                c_start = true_start + rand_offset
                c_end = true_end + abs(rand_offset) if target_len > true_end + abs(rand_offset) else end

            if c_end == target_len:
                c_end = target_len - 1

            total_segments.extend([[c_start, c_end]])

        audio = np.clip(audio, -1, 1)
        total_segments = sorted(total_segments, key=lambda x: x[-1])

        return audio, total_segments

    def get_segments(self):
        pass


class Array:
    __slots__ = ["files", "name"]

    def __init__(self, files):  # files are assumed to be ordered by channel CH1 CH2 etc

        self.files = files
        self.name = files[0].split("/")[-1]

        for i in files:
            assert self.name.split(".CH")[0] == i.split("/")[-1].split(".CH")[0]

    def get_name(self):

        return self.files[0].split("/")[-1].split(".wav")[0].split(".CH")[0].split("_")[-1]

    def get_session(self):

        return self.files[0].split("/")[-1].split("_")[0]

    def read_data(self, boundaries):
        res = []
        for i in range(len(self.files)):
            temp, _ = sf.read(self.files[i], start=boundaries[0], stop=boundaries[1], dtype="float32")
            res.append(temp)

        return np.vstack(res).transpose()

def parse_json(device, json_dir, fs, hp):  # nearest only can be either 0, 1, 2, ..6 k_nearest based on TOF
    '''
    this function parses json file and return segments that belongs to specified device
    if nearest is set under hp devices which are too far from source are discarded for each utterance in jsons
    :param device:
    :param json_dir:
    :param fs:
    :param hp:
    :return:
    '''

    json_file = device.get_session() + ".json"
    json_file = json_dir + "/" + json_file

    with open(json_file, "r") as f:
        data = json.load(f)

    # take the speaker names from first utterance
    speakers_names = [x for x in data[0]["start_time"].keys() if x[0] == "P"]

    diarization = {}
    for spk in speakers_names:
        diarization[spk] = []  # attempt to build diarization from alignment + json

    for utt in data:

        if utt["words"] == "[redacted]" or not utt["words"]:
            continue  # do not include in training examples redacted portions

        # not redacted --> if device is in the utterance take the sads

        if device.get_name() in utt["start_time"].keys():  # else ignore the device

            if not utt["alignments"]:
                continue

            utt["alignments"] = merge_contiguos(utt["alignments"])


            if hp.data.nearest == 0:  # add all devices

                # add the offset
                offset = to_decimal(utt["start_time"][device.get_name()]) - to_decimal(utt["start_time"]["original"])

                for j in range(len(utt["alignments"])):
                    utt["alignments"][j] = [utt["alignments"][j][0] + offset,
                                            utt["alignments"][j][1] + offset]

                diarization[utt["speaker"]].append(utt["alignments"])



    # here we have the diarization for each speaker for the considered device

    tot_segments = []
    flatten = lambda l: [item for sublist in l for item in sublist]  # flatten the list
    for key, elem in diarization.items():  # from multiple speakers to unique list of segments with possible overlap
        # optional smoothing

        if elem:
            temp = flatten(elem)
            if hp.data.smooth:
                temp = sorted(temp, key=lambda x: x[:][-1])  # sort with the end
                temp = smooth_segments(temp, hp.data.collar,
                                       hp.data.th)  # optionally merge segments which are too far apart
            temp = [[int(s * fs), int(e * fs)] for s, e in temp]  # convert to samples
            tot_segments.extend(temp)  # add segments for current speaker

    return sorted(tot_segments, key=lambda x: x[:][1])  # sort them according to end !!



class CHiME5Dataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.target_len = hp.data.audio_len

        if train == True:
            wav_path = hp.data.train_wav_dir
            sad_path = hp.data.train_json_dir
            save_examples = hp.data.train_save_examples

        else:
            wav_path = hp.data.val_wav_dir
            sad_path = hp.data.val_json_dir
            save_examples = hp.data.dev_save_examples

        self.hp = hp

        # now wav_path, xml_path and maf path are lists

        assert len(wav_path) == len(sad_path)

        tot_devices = []
        array_hash = []

        for findx in range(len(wav_path)):

            print("Building hashtabs for Dataloader for dataset in : {}...".format(wav_path[findx]))
            array_hash.append(hashchime(wav_path[findx]))
            # hashtable with all sessions and all files

            ## plan get the total lenght in samples of for all the arrays, then build a list in which each entry points to
            ## example obj with file path and start and stop in samples of the window

            tot_dataset_devices = []

            for sess in array_hash[findx].keys():

                for device in array_hash[findx][sess]:
                    tot_dataset_devices.append(Array(array_hash[findx][sess][device]))

            tot_devices.append(tot_dataset_devices) # build a list of total devices, we are training considering each device separately now

        self.tot_examples = []

        if save_examples:
            try:
                with open(save_examples, "rb") as f:
                    self.tot_examples = pickle.load(f)
                print("Loaded parsed examples from {}".format(save_examples))
                return
            except FileNotFoundError:
                pass

        print("list of examples not found building it now")

        for findx in range(len(wav_path)):


            print("Parsing examples for dataset in {}".format(wav_path[findx]))

            for device in tqdm(tot_devices[findx]):  # tot devices contains list of array objects

                audio_infos = [sf.SoundFile(f) for f in device.files]

                length = len(audio_infos[0])

                # assert same length

                for i in range(1, len(audio_infos)):
                    assert length == len(audio_infos[i])

                starts = np.arange(0, length - hp.data.audio_len + 1, hp.data.stride_length)

                stops = np.arange(hp.data.audio_len, length + 1, hp.data.stride_length)

                sads = parse_json(device, sad_path[findx], self.hp.data.fs, hp)  # parse JSON file

                for j in range(len(stops)):

                    # sads ideally a list of [[start, stop], [etc ]

                    # check current start

                    c_segments = []

                    for s, e in sads:  # critical !!!
                        # segments are already sorted
                        if starts[j] <= e and s < stops[j]:  # overlap

                            s = s - starts[j] if s - starts[j] > 0 else 0
                            e = e - starts[j] if e - starts[
                                j] <= hp.data.audio_len - 1 else hp.data.audio_len - 1  # relative position in samples in current window

                            c_segments.append([s, e])

                        #f s > stops[j]:  # segments are sorted no need to go through all them !!

                         #   break

                    if len(c_segments) > 0:
                        self.tot_examples.append(
                            Example(device, boundaries=(starts[j], stops[j]), segments=c_segments))

        # tot examples contains both path to wave and segments

        if save_examples:
            with open(save_examples, "wb") as f:
                pickle.dump(self.tot_examples, f)
            print("Saved parsed examples in {}".format(save_examples))



    def __len__(self):
        return len(self.tot_examples)

    def __getitem__(self, idx):

        # take some random utterance
        audio, segments  = self.tot_examples[idx].read_example()

        n_fft = self.hp.features.n_fft
        hop_length = self.hp.features.hop
        #pad_length = len(audio) + n_fft - hop_length
        gcc_length = 2 * n_fft  # 50 ms instead of 25
        gcc_pad = len(audio) + gcc_length - hop_length
        nfram = self.hp.data.audio_len // self.hp.features.hop
        fs = self.hp.data.fs

        # Kinect device params
        ngrid = 181
        mic_pos = np.array([-.113, -.076, -.036, .113])

        stfts = [None]*4 # list contains stfts for mics

        # as suggested by emmanuel compute stft only one time for both gcc-phat and spectrogram
        if self.hp.features.nb_multi != 0:
            if self.hp.features.nb_multi == 1:
                # extreme pair only
                pairs = np.array([[0, 3]])

                stfts[0] = librosa.core.stft(librosa.util.pad_center(audio[:, 0], gcc_pad), n_fft=gcc_length,
                                             hop_length=hop_length, center=False)
                stfts[3] = librosa.core.stft(librosa.util.pad_center(audio[:, 3], gcc_pad), n_fft=gcc_length,
                                               hop_length=hop_length, center=False)

            elif self.hp.features.nb_multi == 3:
                # All left-right pairs
                pairs = np.array([[0, 3], [1, 3], [2, 3]])

                for mic in range(4):
                    chl = audio[:, mic]
                    stfts[mic] = librosa.core.stft(librosa.util.pad_center(chl, gcc_pad), n_fft=gcc_length,
                                             hop_length=hop_length, center=False)


            elif self.hp.features.nb_multi == 4:
                # All pairs
                pairs = np.array([[0, 3], [1, 3], [2, 3], [0, 2], [1, 2], [0, 1]])

                for mic in range(4):
                    chl = audio[:, mic]
                    stfts[mic] = librosa.core.stft(librosa.util.pad_center(chl, gcc_pad), n_fft=gcc_length,
                                             hop_length=hop_length, center=False)

            else:
                raise EnvironmentError("Invalid value for nb_multi")

            gcc_spec = np.zeros((ngrid, nfram))
            # compute gcc-phat with cached stfts
            for p in range(len(pairs)):
                chl_stft = stfts[pairs[p, 0]]
                #chl_stft = librosa.core.stft(librosa.util.pad_center(chl, gcc_pad), n_fft=gcc_length,
                 #                            hop_length=hop_length, center=False)
                [nbin, nfram] = chl_stft.shape
                chr_stft = stfts[pairs[p, 1]]

                mic_distance = mic_pos[pairs[p, 1]] - mic_pos[pairs[p, 0]]
                freq_bins = np.arange(nbin) * fs / gcc_length
                freq_bins = np.expand_dims(freq_bins, 1)
                freq_bins = freq_bins.transpose(1, 0)
                tau_grid = np.linspace(-mic_distance / 340, mic_distance / 340, ngrid)
                tau_grid = np.expand_dims(tau_grid, 1)
                projection_matrix = np.exp(-2 * 1j * np.pi * tau_grid @ freq_bins)
                phase_diff = chl_stft * np.conj(chr_stft)
                phase_diff = phase_diff / np.abs(phase_diff + np.finfo(np.float).eps)
                gcc_spec = gcc_spec + np.real(projection_matrix @ phase_diff)


            # concatenate with single-channel features
            # normalize
            sc_feat = stfts[self.hp.features.nb_mono]
            sc_feat = np.abs(sc_feat) #np.sqrt(sc_feat.imag**2 + sc_feat.real**2)
            #sc_feat = (sc_feat - np.mean(sc_feat)) / (np.var(sc_feat) + np.finfo(np.float).eps)

            feat = np.concatenate((sc_feat, gcc_spec))

        else:

            feat = librosa.core.stft(librosa.util.pad_center(audio[:, self.hp.features.nb_mono], gcc_pad), n_fft=gcc_length,
                                             hop_length=hop_length, center=False)

            feat = np.abs(feat) #np.sqrt(feat.imag**2 + feat.real**2)
            #feat = (feat-np.mean(feat)) / (np.var(feat) + np.finfo(np.float).eps)

        feat = torch.from_numpy(feat.astype("float32")) # audio will be processed in gpu with torch.stft
        label = torch.zeros((5, nfram))
        label[0, :] = 1

        for s, e in segments:
            # start
            start_frame, r = divmod(int(s), self.hp.features.hop)  # hop length
            if start_frame == nfram:
                continue
            speech_ratio = (self.hp.features.hop - r) / self.hp.features.hop  # percentage of speech in current frame
            label_prev = label[0:4, start_frame].clone()
            label[0:4, start_frame] = (1 - speech_ratio) * label_prev
            label[1:5, start_frame] += speech_ratio * label_prev
            # stop
            stop_frame, r = divmod(int(e), self.hp.features.hop)
            if stop_frame == nfram:  # if it goes out of boundaries
                r = 0
                stop_frame = stop_frame - 1
            speech_ratio = (r + 1) / self.hp.features.hop  # percentage of speech in current frame
            label_prev = label[0:4, stop_frame].clone()
            label[0:4, stop_frame] = (1 - speech_ratio) * label_prev
            label[1:5, stop_frame] += speech_ratio * label_prev
            # middle
            label[1:5, start_frame + 1:stop_frame] = label[0:4, start_frame + 1:stop_frame].clone()
            label[0, start_frame + 1:stop_frame] = 0

        return feat, label
