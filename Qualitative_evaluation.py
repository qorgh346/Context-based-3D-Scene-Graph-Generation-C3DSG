import os,sys
import json
from collections import defaultdict
hobeDB = defaultdict(int)
relList = ['hanging on','bigger than','smaller than','higher than','leaning against','cover','part of','lying on','same as']
def process(file):
    with open(file,'r') as rel_json:
        datas = json.load(rel_json)
    idx = 0
    for scan in datas['scans']:
        for id,obj in scan['objects'].items():
            if obj == 'tv':
                for rel in scan['relationships']:

                    if rel[3]: #in relList:
                        key = scan['scan']
                        hobeDB[key] = dict()
                        hobeDB[key]['objectNum'] = len(scan['objects'])
                        hobeDB[key]['objects'] = set(list(scan['objects'].values()))
                        hobeDB[key]['triplet'] = list()
                        print(key)
                        try:
                            subject = scan['objects'][str(rel[0])]
                            predicate = rel[3]
                            object = scan['objects'][str(rel[1])]
                        except:
                            continue
                        hobeDB[key]['triplet'].append('{} {} {}'.format(subject, predicate, object))
                break



    print(hobeDB)

if __name__ == '__main__':
    fileName = 'relationships_validation.json'
    root = os.path.abspath('.')
    filePath = os.path.join(root,'data/hobeNew',fileName)
    process(filePath)