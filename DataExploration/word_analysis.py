# count freq of words
# total number of words
# to be used for word embedding
import os
import operator
from pprint import pprint
from xml.etree.ElementTree import parse

words = {}
chars = {}

def words_in_doc(file_location):
    xml = parse(file_location).getroot()

    for line in xml[1]:
        for word in line:
            if word.tag == 'word':
                if word.attrib['text'] not in words:
                    words[word.attrib['text']] = 0
                words[word.attrib['text']] += 1
                # iterate letters
                for char in word.attrib['text']:
                    if char not in chars:
                        chars[char] = 0
                    chars[char] += 1

def iter_docs(data_dir='../data/xml/'):
    corpus = {}
    for xml_file in os.listdir(data_dir):
        words_in_doc(data_dir + xml_file)

def word_to_char_ids_swap(word, char_ids):
    list_char_ids = []
    for char in word:
        list_char_ids.append(char_ids[char])
    return list_char_ids

def load_char_ids():
    char_ids = {0:'~'}
    with open('../DataExploration/char_frequency.txt','r') as char_freq:
        i = 1 # start at 1; blank char is 0
        for line in char_freq:
            char,_ = line.split('|')
            # char: ID ----- ID: char
            char_ids[char] = i; char_ids[i] = char
            i += 1
    return char_ids

def load_word_ids():
    word_ids = {}
    with open('../DataExploration/word_frequency.txt','r') as word_freq:
        i = 0
        for line in word_freq:
            word,_ = line.split('|')
            word_ids[word] = i
            i += 1

    return word_ids

if __name__ == '__main__':
    iter_docs()

    with open('word_frequency.txt','w') as word_freq:
        sorted_words = sorted(words.items(), key=operator.itemgetter(1),reverse=True)
        for elem in sorted_words:
            word_freq.write(elem[0] + '|' + str(elem[1]) + '\n')

    with open('char_frequency.txt','w') as char_freq:
        sorted_chars = sorted(chars.items(), key=operator.itemgetter(1),reverse=True)
        for elem in sorted_chars:
            char_freq.write(elem[0] + '|' + str(elem[1]) + '\n')

    print('Words:',len(words),'Characters:',len(chars))
