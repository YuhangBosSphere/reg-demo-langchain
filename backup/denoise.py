from typing import List
import re

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_lines(text:str) -> List[str]:
    text = text.replace("\r\n","\n")
    return [ln.strip() for ln in text.split("\n")]

from collections import Counter

def denoise_boilerplate(
    text:str,
    freq_ratio: float = 0.15,
    min_len: int = 6
) -> str:
    lines = split_lines(text)
    nonempty = [ln for ln in lines if ln]

    if len(nonempty) <50:
        return text

    counts = Counter(nonempty)
    threshold = max(2, int(len(nonempty) * freq_ratio))

    cleaned = []
    for ln in lines:
        if not ln:
            cleaned.append("")
            continue
        if len(ln) >= min_len and counts.get(ln,0) >= threshold:
            continue
        if re.fullmatch(r"Page\s*\d+(\s*of\s*\d+)?", ln, flags = re.I):
            continue
        cleaned.append(ln)

    return normalize_whitespace("\n".join(cleaned))

text = """
This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

Page 1

Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10

Page 1

Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Page 1

Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Page 1

Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10
Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10Introduction

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

The purpose of this study is to examine the impact of AI-assisted instruction on middle school mathematics performance.

Page 2 of 10

Methods

This document is confidential and intended solely for the use of the individual or entity to whom it is addressed.

Data were collected from three public schools in the Boston metropolitan area."""

print(denoise_boilerplate(text))
        
