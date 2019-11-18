import numpy as np


def clip(val, min, max):

    if val < min:
        return min
    elif val > max:
        return max
    else:
        return val

def count_overlaps(segments, seglen):


    def helper(segments):

        maxlen= segments[-1][1]

        zeros = np.zeros(maxlen)

        for seg in segments:
            zeros[seg[0]:seg[-1]] += 1

        return zeros


    zeros = helper(segments)
    speech = len(np.where(zeros==1)[0])


    return seglen-speech, speech,  len(np.where(zeros==2)[0]), \
           len(np.where(zeros==3)[0]), len(np.where(zeros==4)[0])



def smooth_segments(segments, collar, th):
    '''
    Note: potential bug: if collar is big enough it can cause the re-segmented segments to go out of bounds at the end.
    Not really a problem in CHiME-5
    :param segments:
    :param collar:
    :param th:
    :return:
    '''

    lenght = segments[-1][
                 -1] + collar
    for indx, elem in enumerate(segments):

        s, e = elem
        s = clip(s - collar, 0, s - collar)
        e = clip(e + collar, e + collar, lenght)

        segments[indx] = [s, e]

        # collar applied now merging segments

        stack = [segments[0]]

        for indx in range(1, len(segments)):

            n_start = segments[indx][0]
            if n_start - stack[-1][1] <= th:
                # merge
                stack[-1][1] = segments[indx][1]  # next stop
                if n_start < stack[-1][0]:
                    stack[-1][0] = n_start
                indx += 1  # advance 2

            else:

                stack.append(segments[indx])

        segments = stack

        return segments


def merge_contiguos(result):

    '''
    merge contigous and overlapping segments
    :param result:
    :return:
    '''

    # first pass merging of segments
    # merge consecutive segments !

    stack = [result[0]]

    for indx in range(1, len(result)):

        n_start = result[indx][0]
        if n_start <= stack[-1][1]:
            # merge
            stack[-1][1] = result[indx][1]  # next stop
            if n_start < stack[-1][0]:
                stack[-1][0] = n_start
            indx += 1  # advance 2

        else:

            stack.append(result[indx])

    result = stack

    return result





