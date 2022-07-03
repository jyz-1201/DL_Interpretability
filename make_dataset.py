import argparse
import logging
from os import listdir
from os.path import isfile, join
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="all files in this directory will be made into dataset")

    ## Other parameters

    # print arguments
    args = parser.parse_args()
    logger.info(args)
    input_dir = args.input_dir

    files = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]

    for data_file in files:
        with open(data_file, 'r') as file:
            Lines = file.readlines()
            for line in Lines:
                doc = nlp(line)
                print(*[
                    f'{word.text}\t{word.head}\t{word.id}\t{word.deprel}'
                    for sent in doc.sentences for word in sent.words], sep='\n')
                print(*[
                    f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
                    for sent in doc.sentences for word in sent.words], sep='\n')




if __name__ == "__main__":
    main()
