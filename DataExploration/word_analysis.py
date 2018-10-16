# count freq of words
# total number of words
# to be used for word embedding
import os
import operator
from pprint import pprint
from xml.etree.ElementTree import parse

words = {}

def words_in_doc(file_location):
    xml = parse(file_location).getroot()

    for line in xml[1]:
        for word in line:
            if word.tag == 'word':
                if word.attrib['text'] not in words:
                    words[word.attrib['text']] = 0
                words[word.attrib['text']] += 1

def iter_docs(data_dir='../data/xml/'):
    corpus = {}
    for xml_file in os.listdir(data_dir):
        words_in_doc(data_dir + xml_file)

def load_word_ids():
    word_ids = {}
    with open('word_frequency.txt','r') as word_freq:
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

    print(len(words))
