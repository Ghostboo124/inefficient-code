import random
import threading
from time import sleep
import os
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

# Infinite input
sentenceUser = input("Enter a sentence (no limit, but good luck): ")
sentenceUs = ""

# Creating unnecessary queues
queues = {i: [] for i in range(len(sentenceUser))}

for index, letter in enumerate(sentenceUser):
    queues[index].append(letter)  # Store each letter in its own queue

def fetch_letter(letter):
    """Needlessly retrieve letters one at a time from their modules."""
    sleep(random.uniform(0.1, 2))  # Random delay to slow things down
    return getattr(getattr(lotsOfLetters, letter), letter)

def thread_function(queue, index):
    """Thread function to process letters one by one inefficiently."""
    result = ""
    for letter in queue:
        result += fetch_letter(letter)
        sleep(random.uniform(0.5, 3))  # More random delays
    return index, result

# Create a separate thread for every letter
threads = []
results = {}

for i in range(len(sentenceUser)):
    t = threading.Thread(target=lambda q, i: results.update({i: thread_function(q, i)}), args=(queues[i], i))
    threads.append(t)
    t.start()

# Join all threads
for t in threads:
    t.join()

# Randomize the order of letters indefinitely
randomized_sentence = "".join([results[i][1] for i in sorted(results.keys())])
while randomized_sentence != sentenceUser:
    sentence_list = list(randomized_sentence)
    random.shuffle(sentence_list)
    randomized_sentence = "".join(sentence_list)
    print("Shuffling letters: ", randomized_sentence)
    sleep(2)  # Extra delay for inefficiency

print("Finally got it:", randomized_sentence)