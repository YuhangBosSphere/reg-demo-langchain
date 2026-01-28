from typing import List, Dict
def split_lines(text:str) -> List[str]:
    text = text.replace("\r\n","\n")
    return [ln.strip() for ln in text.split("\n")]
from typing import List
import re

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
                            
_heading_md = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_heading_num = re.compile(r"^(\d+(\.\d+)*)\s+(.+?)\s*$")
_heading_colon = re.compile(r"^(.+?):\s*$")

def sectionize(text:str) -> List[Dict]:
    lines = split_lines(text)
    sections = []
    path: List[str] = []
    cur_title = "root"
    cur_body: List[str] = []

    def push_section(title,body_lines, path_snapshot):
        body = normalize_whitespace("\n".join(body_lines))
        if body:
            sections.append({
                "path" : path_snapshot[:],
                "title" : title,
                "body" : body
                })
    for ln in lines:
        m1 = _heading_md.match(ln)
        m2 = _heading_num.match(ln)
        m3 = _heading_colon.match(ln)

        if m1 or m2 or m3:
            push_section(cur_title, cur_body, path)
            cur_body = []
        if m1:
            level = len(m1.group(1))
            title = m1.group(2).strip()
            path = path[:level - 1]
            path.append(title)
            cur_title = title
            continue

        if m2:
            depth = m2.group(1).count(".") + 1
            title = m1.group(3).strip()
            path = path[:depth - 1]
            path.append(title)
            cur_title = title
            continue

        if m3 and len(ln) <= 80:
            title = m3.group(1).strip()
            path.append(title)
            cur_title = title
            continue

        cur_body.append(ln)

    push_section(cur_title, cur_body, path)
    return sections 
        
#append.body -> title -> body[]-> path -> cur_title            
            
            
