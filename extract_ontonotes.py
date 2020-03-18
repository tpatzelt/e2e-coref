import argparse, os, re, csv
import random
from pathlib import Path

# Check if directory exists
def chkdir(chk):
    if not os.path.isdir(chk):
        raise NotADirectoryError(string)
    else:
        return chk

def find_coref_files(directory):
    return list(Path(directory).rglob('*.coref'))

class GapStyleWriter:
    def __init__(self, path):
        self.path = path

    def write_data(self, data, prefix, dest):
        path = os.path.join(self.path, dest)
        with open(path, 'w+') as f:
            # write head
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'A-coref', 'B', 'B-offset', 'B-coref', 'URL'])
            # write rows
            for i, instance in enumerate(data):
                raw, pronoun, A, A_coref, B, B_coref, url = instance
                A_offset = A['start']
                B_offset = B['start']

                tsv_writer.writerow([prefix+'-'+str(i), raw.strip(), pronoun['raw'], pronoun['start'], A['raw'], A_offset, A_coref, B['raw'], B_offset, B_coref, url])

    def write_train(self, data):
        self.write_data(data, 'train', 'ontonotes-development.tsv')

    def write_test(self, data):
        self.write_data(data, 'test', 'ontonotes-test.tsv')

    def write_valid(self, data):
        self.write_data(data, 'validation', 'ontonotes-validation.tsv')
          
# parser to parse .coref inline-annotated files
class CorefParser:
    def __init__(self, path, debug=False):
        self.path = path
        self.debug = debug

    def parse_doc(self, line):
        rgx = re.compile("""\<DOC\sDOCNO="(.*)"\>""")
        match = rgx.match(line)
        if not match:
            raise Exception("ParseError: DOC-tag invalid in: {:s}".format(self.path))
        return match.group(1)

    def is_text_open(self, line):
        rgx = re.compile("""\<TEXT\sPARTNO="(.*)"\>""")
        return rgx.match(line)

    def parse_text(self, line):
        rgx = re.compile("""\<TEXT\sPARTNO="(.*)"\>""")
        match = rgx.match(line)
        if not match:
            raise Exception("ParseError: TEXT-tag invalid in: {:s}".format(self.path))
        return match.group(1)

    def extract_corefs(self, line):
        rgx_start = re.compile("""<COREF\s+ID="(.*)"\s+TYPE="([IDENT|APPOS]+)".*\>""")
        rgx_end = re.compile("""<\/COREF>""")
        corefs = []
        raw = '' 
        idx = 0
        stack = []
        while idx < len(line):
            char = line[idx]
            if char == '<':
                # open
                start = idx
                idx = line.index('>', idx)
                end = idx+1
                content = line[start:end]
                match = rgx_start.match(content)
                if match:
                    # extract
                    stack.append( {'id': match.group(1),
                                   'type': match.group(2),
                                   'start': len(raw)} )
                match = rgx_end.match(content)
                if match:
                    # extract
                    coref = stack.pop()
                    coref['end'] = len(raw)+1
                    coref['raw'] = raw[coref['start']:coref['end']]
                    corefs.append(coref)
            else:
                raw += char
            idx += 1


        if self.debug:
            for coref in corefs:
                print(coref)
        return corefs, raw

    def parse_line(self, line):
        corefs, raw = self.extract_corefs(line)

        return (corefs, raw)

    def is_text_close(self, line):
        rgx = re.compile("""\<\/TEXT\>""")
        return rgx.match(line)

    def parse(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
            doc = self.parse_doc(lines[0])
            lines = lines[1:-1]
            
            texts = []
            currText = None
            for i, line in enumerate(lines):
                if self.is_text_open(line):
                    currText = {'part': self.parse_text(line), 'lines': []}
                elif self.is_text_close(line):
                    texts.append(currText)
                    currText = None
                else:
                    try:
                      parsed = self.parse_line(line)
                      currText['lines'].append(parsed)
                    except Exception:
                        print("\t\tError Parsing Line {:d}: Bad Format".format(i))
            return {'document': doc, 'path': self.path, 'texts': texts}

# parse the coref file
def parse_coref_file(path):
    with open(path) as f:
        parser = CorefParser(path)
        return parser.parse()

parser = argparse.ArgumentParser(description='Extract English Ontonotes data and transform it into format for Referential Reader')
parser.add_argument('input_dir', metavar='IN', type=chkdir, default='./ontonotes',
                    help='path to ontonotes directory')
parser.add_argument('output_dir', metavar='OUT', type=chkdir, default='./refreader/data/ontonotes',
                    help='path to the output directory (refreader/data/ontonotes)')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

annotations_dir = chkdir(os.path.join(input_dir, "ontonotes_partial_english/english/annotations"))
corpora = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
corpora_dirs = [chkdir(os.path.join(annotations_dir, cp)) for cp in corpora]

print("Extracting Ontonotes data from: {:s}".format(args.input_dir))

documents = []
for corpus in corpora_dirs:
    coref_files = find_coref_files(corpus)
    print("\tFound {:5d} documents in {:s}".format(len(coref_files), corpus))
    for f in coref_files:
        doc = parse_coref_file(f)
        documents.append(doc)

print("Merging and filtering dataset")
# merging all documents to one list of instances
instances = []
for doc in documents:
    texts = doc['texts']
    for text in texts:
        for line in text['lines']:
            # (coref_list, raw_line, document_id, text_id)
            instance = (line[0], line[1], doc['document'], text['part'])
            instances.append(instance)

# filtering: only instances with exactly two IDENT-coreferences stay
allowed_pronouns = {'she', 'her', 'hers', 'he', 'his', 'him'}
filtered_instances = []
for instance in instances:
    corefs = instance[0]
    # find 3 corefs: A, B and pronoun. The id of pronoun must match with either A or B
    # first, check if pronoun is under corefs
    pronoun = None
    for coref in corefs:
        if coref['raw'].lower() in allowed_pronouns:
            pronoun = coref
    if not pronoun:
        continue

    # second, check if pronoun with same id exists
    A, B = random.choices(corefs, k=2)
    A_coref = A['id'] == pronoun['id']
    B_coref = B['id'] == pronoun['id']

    # (raw_sentence, pronoun, correct, incorrect, id)
    filtered_instances.append( (instance[1], pronoun, A, A_coref, B, B_coref, instance[2] + instance[3]) )

len_dataset = len(filtered_instances)
print("\tAfter filtering, got total of {:d} instances (was {:d})".format(len_dataset, len(instances)))

# shuffle
random.shuffle(filtered_instances)

# split into TRAIN, TEST and VALIDATION datasets
perc_train = .8
perc_test = .15
perc_valid = .05

len_train = int(len_dataset * perc_train)
len_test = int(len_dataset * perc_test) 
len_valid = len_dataset - len_train - len_test

data_train = filtered_instances[:len_train]
data_test = filtered_instances[len_train:len_test]
data_valid = filtered_instances[len_test:]

print("Split dataset into:")
print("\t{:5d} = {:2.1f}% TRAIN data".format(len_train, perc_train*100))
print("\t{:5d} = {:2.1f}% TEST  data".format(len_test, perc_test*100))
print("\t{:5d} = {:2.1f}% VALID data".format(len_valid, perc_valid*100))

# write to disk
print("Writing formatted data to: {:s}".format(args.output_dir))
writer = GapStyleWriter(output_dir)
writer.write_train(data_train)
writer.write_test(data_test)
writer.write_valid(data_valid)
