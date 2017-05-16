import collections

class WordMap():
    def __init__(self):
        self._wordlist = []
        self.wordmap = {}

    def build(self, words, size):
        self.wordmap = {}
        self._wordlist = []
        # unknown: 0
        for word, _ in collections.Counter(words).most_common(size - 1):
            self._wordlist.append(word)
            self.wordmap[word] = len(self.wordmap) + 1

    def load(self, filename):
        with open(filename, "r") as file:
            self._wordlist = [line.rstrip() for line in file.readlines()]
        self.wordmap = {}
        for word in self._wordlist:
            self.wordmap[word] = len(self.wordmap) + 1

    def save(self, filename):
        with open(filename, "w") as file:
            for word in self._wordlist:
                file.write(word)
                file.write('\n')

    def lookup(self, word):
        return self.wordmap.get(word, 0)

    def size(self):
        return len(self.wordmap) + 1
