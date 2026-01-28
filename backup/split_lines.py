from typing import List
def split_lines(text:str) -> List[str]:
    text = text.replace("\r\n","\n")
    return [ln.strip() for ln in text.split("\n")]

print(split_lines("late arrival\nhaii"))


                                    
