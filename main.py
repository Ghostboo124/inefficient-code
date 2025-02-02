"""
A Really inefficient way of putting together a sentence
"""

# Imports
import random
import threading
from time import sleep
import os

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

# Threads
from threadCode import thread0
from threadCode import thread1
from threadCode import thread2
from threadCode import thread3


if os.cpu_count() <= 4:
    print("You need more than 4 threads!")
elif os.cpu_count() >= 32:
    print("What CPU are you running, that is more than 32 threads!")
else:
    print("Your CPU is supported")

sentenceUser = input("What is the sentence you want to make (no more than 32 chars): ")
sentenceUs = ""

thread0Return = ""
que0 = []

thread1Return = ""
que1 = []

thread2Return = ""
que2 = []

thread3Return = ""
que3 = []

counter = 1
for letter in sentenceUser:
    if counter == 1 or counter == 5 or counter == 9 or counter == 13 or counter == 17 or counter == 21 or counter == 25 or counter == 29:
        que0.append(letter)
    if counter == 2 or counter == 6 or counter == 10 or counter == 14 or counter == 18 or counter == 22 or counter == 26 or counter == 30:
        que1.append(letter)
    if counter == 3 or counter == 7 or counter == 11 or counter == 15 or counter == 19 or counter == 23 or counter == 27 or counter == 31:
        que2.append(letter)
    if counter == 4 or counter == 8 or counter == 12 or counter == 16 or counter == 20 or counter == 24 or counter == 38 or counter == 32:
        que3.append(letter)

    if counter not in range(1,32):
        print("You weren't supposed to enter a sentence with more than 32 characters!!!")
        exit(1)
    counter += 1








class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


"""
print(thread0(que0,sentenceUs))
print(thread1(que1,sentenceUs))
print(thread2(que2,sentenceUs))
print(thread3(que3,sentenceUs))
"""

print("About to create")
t0 = ThreadWithReturnValue(target=thread0.thread0Letters, args=(que0,sentenceUs,))
t1 = ThreadWithReturnValue(target=thread1.thread1Letters, args=(que1,sentenceUs,))
t2 = ThreadWithReturnValue(target=thread2.thread2Letters, args=(que2,sentenceUs,))
t3 = ThreadWithReturnValue(target=thread3.thread3Letters, args=(que3,sentenceUs,),)

print("Created!")
print("About to start")

t0.start()
t1.start()
t2.start()
t3.start()

print("Started!")
print("About to join")

thread0Return = t0.join()
thread1Return = t1.join()
thread2Return = t2.join()
thread3Return = t3.join()

print("Joined!")
os.system("pause")

sentenceUs = thread0Return + thread1Return + thread2Return + thread3Return
sentenceUsRandom = sentenceUs

if sentenceUs == sentenceUser:
    print("Uhh this should be impossible")
elif sentenceUs != sentenceUser:
    print("About to start randomising the generated sentence")
    while sentenceUsRandom != sentenceUser:
        sentenceUsRandomList = list(sentenceUsRandom)
        random.shuffle(sentenceUsRandomList)
        sentenceUsRandom = ''.join(sentenceUsRandomList)
        print("    Random Sentence: " + sentenceUsRandom)
    sentenceUsRandom = sentenceUs
    print("Got It!", end=": ")

print(sentenceUs)