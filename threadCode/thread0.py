# Imports
from time import sleep

# Letters
import lotsOfLetters.a
import lotsOfLetters.b
import lotsOfLetters.c
import lotsOfLetters.d
import lotsOfLetters.e
import lotsOfLetters.f
import lotsOfLetters.g
import lotsOfLetters.h
import lotsOfLetters.i
import lotsOfLetters.j
import lotsOfLetters.k
import lotsOfLetters.l
import lotsOfLetters.m
import lotsOfLetters.n
import lotsOfLetters.o
import lotsOfLetters.p
import lotsOfLetters.q
import lotsOfLetters.r
import lotsOfLetters.s
import lotsOfLetters.t
import lotsOfLetters.u
import lotsOfLetters.v
import lotsOfLetters.w
import lotsOfLetters.x
import lotsOfLetters.y
import lotsOfLetters.z

# Code

def thread0Letters(que: list, sentence: str) -> str:
    """
    The script for thread 0
    - que:      The que that is to be read from
    - sentence: The sentence to put the output into
    """
    returnSentence = ""
    for i in que:
        if i == "a":
            returnSentence += lotsOfLetters.a.a
        if i == "b":
            returnSentence += lotsOfLetters.b.b
        if i == "c":
            returnSentence += lotsOfLetters.c.c
        if i == "d":
            returnSentence += lotsOfLetters.d.d
        if i == "e":
            returnSentence += lotsOfLetters.e.e
        if i == "f":
            returnSentence += lotsOfLetters.f.f
        if i == "g":
            returnSentence += lotsOfLetters.g.g
        if i == "h":
            returnSentence += lotsOfLetters.h.h
        if i == "i":
            returnSentence += lotsOfLetters.i.i
        if i == "j":
            returnSentence += lotsOfLetters.j.j
        if i == "k":
            returnSentence += lotsOfLetters.k.k
        if i == "l":
            returnSentence += lotsOfLetters.l.l
        if i == "m":
            returnSentence += lotsOfLetters.m.m
        if i == "n":
            returnSentence += lotsOfLetters.n.n
        if i == "o":
            returnSentence += lotsOfLetters.o.o
        if i == "p":
            returnSentence += lotsOfLetters.p.p
        if i == "q":
            returnSentence += lotsOfLetters.q.q
        if i == "r":
            returnSentence += lotsOfLetters.r.r
        if i == "s":
            returnSentence += lotsOfLetters.s.s
        if i == "t":
            returnSentence += lotsOfLetters.t.t
        if i == "u":
            returnSentence += lotsOfLetters.u.u
        if i == "v":
            returnSentence += lotsOfLetters.v.v
        if i == "w":
            returnSentence += lotsOfLetters.w.w
        if i == "x":
            returnSentence += lotsOfLetters.x.x
        if i == "y":
            returnSentence += lotsOfLetters.y.y
        if i == "z":
            returnSentence += lotsOfLetters.z.z
        sleep(1)
    sentence += returnSentence
    return returnSentence