def to_decimal(x):
    hours, mins, sec = x.split(":")
    hours = int(hours)
    mins = int(mins)
    sec, dec = sec.split(".")
    sec = int(sec)
    dec = int(dec)

    return hours * 3600 + mins * 60 + sec + dec * 0.01