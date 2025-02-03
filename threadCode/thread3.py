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
import lotsOfLetters.one
import lotsOfLetters.two
import lotsOfLetters.three
import lotsOfLetters.four
import lotsOfLetters.five
import lotsOfLetters.six
import lotsOfLetters.seven
import lotsOfLetters.eight
import lotsOfLetters.nine
import lotsOfLetters.exclamation
import lotsOfLetters.question
import lotsOfLetters.stop
import lotsOfLetters.space

# Code

def thread3Letters(que: list, sentence: str) -> str:
    """
    The script for thread 3
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
        if i == "1":
            returnSentence += lotsOfLetters.one.one
        if i == "2":
            returnSentence += lotsOfLetters.two.two
        if i == "3":
            returnSentence += lotsOfLetters.three.three
        if i == "4":
            returnSentence += lotsOfLetters.four.four
        if i == "5":
            returnSentence += lotsOfLetters.five.five
        if i == "6":
            returnSentence += lotsOfLetters.six.six
        if i == "7":
            returnSentence += lotsOfLetters.seven.seven
        if i == "8":
            returnSentence += lotsOfLetters.eight.eight
        if i == "9":
            returnSentence += lotsOfLetters.nine.nine
        if i == ".":
            returnSentence += lotsOfLetters.stop.stop
        if i == "!":
            returnSentence += lotsOfLetters.exclamation.exclamation
        if i == "?":
            returnSentence += lotsOfLetters.question.question
        if i == " ":
            returnSentence += lotsOfLetters.space.space
        # if i not in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"] or i not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] or i not in [" ", ".", "?", "!"]:
        #     returnSentence += lotsOfLetters.space.space
        sleep(1)
    sentence += returnSentence
    thread3Return = returnSentence
    return returnSentence
