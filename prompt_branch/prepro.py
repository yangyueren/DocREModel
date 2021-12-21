from tqdm import tqdm
from transformers.models.auto.tokenization_auto import logger
import ujson as json
import numpy as np
import pickle as pkl
import os
import copy

from torch.utils.data import Dataset
docred_rel2id = json.load(open('./dataset/docred/DocRED_baseline_metadata/rel2id.json', 'r'))
docred_rel2id['Na'] = 0
docred_rel2info = json.load(open('./dataset/docred/rel_info.json', 'r'))
docred_rel2info['Na'] = 'nothing'
docred_id2relinfo = {docred_rel2id[k]: (v.split(' ') + ['[unused1]']*4)[:4] for k,v in docred_rel2info.items()}

# from ast import literal_eval
# ner_pair2rel = json.load(open('./common/ner_pair2rel.json', 'r'))
# ner_pair2rel = {literal_eval(k): v for k, v in ner_pair2rel.items()}

for k,v in docred_id2relinfo.items():
    v = [f'[{i}]'for i in v]
    docred_id2relinfo[k] = v

cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


class Verbalizer:
    def __init__(self, docred_rel2id, tokenizer) -> None:
        self.relid2token = {}
        self.tokenizer = tokenizer
        for k,v in docred_rel2id.items():
            token = f'[unused{v}]'
            self.relid2token[v] = token
        tokens = []
        for k in sorted(self.relid2token.keys()):
            tokens.append(self.relid2token[k])
        special_tokens_dict = {'additional_special_tokens': tokens}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f'add {num_added_toks} specialtokens, they are {tokenizer.additional_special_tokens}')

        self.tokenids = self.tokenizer.convert_tokens_to_ids(tokens)

    def get_token_of_rel(self, relid):
        return self.relid2token[relid]

    def get_all_verbalizer_tokenids(self):
        return self.tokenids

    def get_tokenid_of_relid(self, relids):
        tokens = [self.relid2token[relid] for relid in relids]
        tokenids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokenids

class DocRedDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length, name) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pos_samples = 0
        self.neg_samples = 0
        self.feature_dict = {}
        self.name = name
        logger.info(f'Init DocRedDataset {name}')

    def __getitem__(self, index):
        if index in self.feature_dict:
            # print('hit', index, self.name, len(self.feature_dict))
            return copy.deepcopy(self.feature_dict[index])
        # print('unhit', index, self.name, len(self.feature_dict))
        tokenizer, max_seq_length = self.tokenizer, self.max_seq_length
        sample = self.data[index]
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
                tokens_wordpiece = tokenizer.tokenize(token)
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
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        
        # docred_prompt
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
            relations.append(relation)
            hts.append([h, t])
            if mention["relation"] > 0:
                self.pos_samples += 1
            else:
                self.neg_samples += 1
        ''' docred
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1
        '''


        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   }
        self.feature_dict[index] = feature
        return copy.deepcopy(feature)

    def __len__(self):
        return len(self.data)


def read_docred(file_in, tokenizer, max_seq_length=1024, sample_ratio=1.0):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    
    if 'pkl' in file_in:
        if os.path.exists(file_in+str(sample_ratio)):
            data = pkl.load(open(file_in+str(sample_ratio), 'rb'))
        else:
            with open(file_in, "rb") as fh:
                data = pkl.load(fh)
                data = data[:int(len(data)*sample_ratio)]
            pkl.dump(data, open(file_in+str(sample_ratio), 'wb'))
    else:
        with open(file_in, "r") as fh:
            data = json.load(fh)
            data = data[:int(len(data)*sample_ratio)]
    return DocRedDataset(data, tokenizer=tokenizer, max_seq_length=max_seq_length, name=file_in)

    """
    for sample in tqdm(data, desc="Example"):
        # import pdb;pdb.set_trace()
        if sample['labels'][0]['r'] == 'Na' and np.random.random() > 0.1 and 'train' in file_in:
            # neg_samples -= 1
            # import pdb;pdb.set_trace()
            continue


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
                tokens_wordpiece = tokenizer.tokenize(token)
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
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        
        # docred_prompt
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
            relations.append(relation)
            hts.append([h, t])
            if mention["relation"] > 0:
                pos_samples += 1
            else:
                neg_samples += 1
        ''' docred
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1
        '''


        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

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
    """




"""
def read_docred(file_in, tokenizer, max_seq_length=512):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)[:]

    infos = set()
    # import pdb; pdb.set_trace()
    for rel, info in docred_id2relinfo.items():
        for i in info:
            infos.add(i)
    infos = list(infos) + ['[unused1]',]
    special_tokens_dict = {'additional_special_tokens': infos}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f'add {num_added_toks} specialtokens, they are {tokenizer.additional_special_tokens}')
    ids = sorted(tokenizer.additional_special_tokens_ids)
    mask_token2id = {}
    for i in range(len(ids)):
        mask_token2id[ids[i]] = i

    for sample in tqdm(data, desc="Example"):
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
                tokens_wordpiece = tokenizer.tokenize(token)
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
                r = int(docred_rel2id[label['r']])
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
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        if 'train' in file_in:
            for i in range(len(relations)):
                rel = relations[i]
                rel_id = np.where(np.array(rel) > 0)[0][-1]
                if rel_id == 0 and np.random.random() > 0.9:
                    continue

                h,t = hts[i]
                h_name = entities[h][0]['name']
                t_name = entities[t][0]['name']
                info = docred_id2relinfo[rel_id]
                # import pdb;pdb.set_trace()
                masks = ['[MASK]']*len(info)
                # import pdb;pdb.set_trace()
                sentd = f'From the paragraph , we can conclude that {h_name} and {t_name} has relation of {" ".join(masks)} . '.split(' ') + sents
                sentd = sentd[:max_seq_length - 2]
                input_ids = tokenizer.convert_tokens_to_ids(sentd)
                
                input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

                mask_ids = tokenizer.convert_tokens_to_ids(info)
                mask_ids = [mask_token2id[tokenid] for tokenid in mask_ids]

                i_line += 1
                feature = {'input_ids': input_ids,
                        'entity_pos': entity_pos,
                        'labels': relations,
                        'hts': hts,
                        'title': sample['title'],
                        'mask_ids': mask_ids,
                        }
                
                features.append(feature)

        else:

            sents = sents[:max_seq_length - 2]
            
            input_ids = tokenizer.convert_tokens_to_ids(sents) # [115, 163, 2556, 14099]
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids) #[101, 115, 163, 2556, 14099, 102] add cls and sep

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
"""

def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features