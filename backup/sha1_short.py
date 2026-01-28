import hashlib
def sha1_short(s:str, n:int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors = "ignore")).hexdigest()[:n]



print(sha1_short("late arrival"))

