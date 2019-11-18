import fnmatch
import os


def get_files(dir, pattern="*.wav", recursive=False):
    '''Recursively finds all files matching the pattern in a specific directory tree'''

    if not os.path.isdir(dir):
        raise IOError("path provided is not a directory")

    files = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

        if recursive == False:
            break

    return files
