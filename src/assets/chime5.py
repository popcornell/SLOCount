import fnmatch
import os

def hashchime(chime5dir):

    hash_sess = {}

    pattern = "*U*.wav"

    if not os.path.isdir(chime5dir):
        raise IOError("path provided is not a directory")

    files = []
    for root, dirnames, filenames in os.walk(chime5dir):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))


    for file in files:

        session = file.split("/")[-1].split("_")[0]
        device = file.split("/")[-1].split("_")[1].split(".")[0]

        if session not in hash_sess.keys():

            hash_device = {}
            hash_device[device] = [file]

            hash_sess[session] = hash_device

        else:

            if device not in hash_sess[session].keys():

                hash_sess[session][device] = [file]

            else:

                hash_sess[session][device].append(file)

    for sess in hash_sess.keys():

        for device in hash_sess[sess]:

            hash_sess[sess][device] = sorted(hash_sess[sess][device], key=lambda x: x.split("/")[-1].split(".wav")[0].split(".CH")[-1])

    return hash_sess