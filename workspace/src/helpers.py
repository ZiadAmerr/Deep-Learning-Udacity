import unicodedata
import re



######################################################################
#                                                                    #
# Acquired from pytorch.org/tutorials/beginner/chatbot_tutorial.html #
#                                                                    #
######################################################################

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        # Initializes empty vocab
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
    
    def addSentence(self, sentence):
        # add sentence by looping through each word and adding it
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        
        # If the word IS NOT in the indexed dict
        if word not in self.word2index:
            # Add it with index = num_words (len(words)+1)
            self.word2index[word] = self.num_words
            
            # with freq=1
            self.word2count[word] = 1
            
            # reverse-index it
            self.index2word[self.num_words] = word
        
            # increment num_words
            self.num_words += 1
            
            # Since we are adding a word, then the new vocab is not trimmed
            self.trimmed = False
            
        else:
            # if the word is IN the indexed dict, increment freq by 1
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        # Check if Voc is already trimmed, if it is, terminate
        if self.trimmed:
            return

        # empty list of words we are keeping
        keep_words = []

        # for each key(word), value(count) in word2count.items()
        for k, v in self.word2count.items():
            # if value(count) >= threshold
            if v >= min_count:
                # then add that key(word) to our kept words
                keep_words.append(k)

        # Print info about kept words
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        # Now let's add all the words again
        for word in keep_words:
            self.addWord(word)
        
        # If process successful, trimmed flag should be toggled True
        self.trimmed = True

        

MAX_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def readVocs(dataframe, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = []
    pairs = []
    for i, row in dataframe.iterrows():
        qst = normalizeString(row["question"])
        ans = normalizeString(row["answer"])
        lines.append(qst)
        lines.append(ans)
        pairs.append([qst, ans])
      
    # Split every line into pairs and normalize
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]