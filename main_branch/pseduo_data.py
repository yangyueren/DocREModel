import os
import json
from typing_extensions import get_origin
import tqdm
import copy

folder_path = './data/DocRED/'
error_num = 0
total_num = 0
tick_num = 0

def one_data(sample):
    
    s = copy.deepcopy(sample)
    # for m in copy.deepcopy(sample['labels']):

    #     if m['h'] not in m['evidence'] or m['t'] not in m['evidence'] :
    #         global tick_num
    #         tick_num += 1

    ans = []
    for title_idx, label in enumerate(copy.deepcopy(sample['labels'])):
        global total_num
        total_num += 1
        evidence = label['evidence']
        newsents = []
        oldsent2newsent = {}
        for i in evidence:
            oldsent2newsent[i] = len(newsents)
            newsents.append(copy.deepcopy(sample['sents'][i]))
        
        tmpvertexSet = []
        for v in copy.deepcopy(sample['vertexSet']):
            mentions = [i for i in v if i['sent_id'] in evidence]
            tmpvertexSet.append(mentions)
        # import pdb; pdb.set_trace()
        
        oldver2newver = {}
        newvertexSet = []
        for idx, v in enumerate(tmpvertexSet):
            if len(v) != 0:
                oldver2newver[idx] = len(newvertexSet)
                newvertexSet.append(v)
        
        for entity in newvertexSet:
            for m in entity:
                assert sample['sents'][m['sent_id']] == newsents[oldsent2newsent[m['sent_id']]], 'sent error'
                m['sent_id'] = oldsent2newsent[m['sent_id']]
        
        try:
            assert len(tmpvertexSet[label['h']]) > 0, 'error'
            assert len(tmpvertexSet[label['t']]) > 0, 'error'
            
            label['h'] = oldver2newver[label['h']]
            label['t'] = oldver2newver[label['t']]
            label['evidence'] = [oldsent2newsent[i] for i in label['evidence']]

            piece = {
                'title': sample['title'],
                'vertexSet': newvertexSet,
                'labels': [label],
                'sents': newsents
            }

            for entity in piece['vertexSet']:
                for m in entity:
                    try:
                        # assert set(m['name']) == set(' '.join(piece['sents'][m['sent_id']][m['pos'][0]: m['pos'][1]])), 'validate error'
                        assert len(piece['sents'][m['sent_id']]) >= m['pos'][1], 'validate error'
                    except Exception as e:
                        import pdb;
                        pdb.set_trace()
        
            ans.append(piece)
            # debug
            break

        except Exception as e:
            # import pdb; pdb.set_trace()
            # print(title_idx, e)
            
            global error_num
            error_num += 1
    # print(len(ans))
    assert s == sample, 'changed'
    
    
    return ans


def process_file(file_name, output_name):
    path = os.path.join(folder_path, file_name)
    with open(path, "r") as fh:
        data = json.load(fh)
    ans = []
    for sample in data:

        cur = one_data(sample)
        ans += cur
        
    with open(os.path.join(folder_path, output_name), 'w') as f:
        json.dump(ans, f)
    
process_file('train_annotated.json', 'train_annotated_only_evidence.json')
process_file('dev.json', 'dev_only_evidence.json')
print(error_num, total_num, error_num/total_num)


def read_docred(file_in, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in data:
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = [token]
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = 1
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                try:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))
                except Exception as e:
                    import pdb; pdb.set_trace()
                    print(e)

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * 97
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (97 - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = sents

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features

def test():
    
    train_file = os.path.join(folder_path, 'train_annotated_only_evidence.json')
    # train_file = os.path.join(folder_path, 'train_annotated.json')
    read_docred(train_file)

test()
