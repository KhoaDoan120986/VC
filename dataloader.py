from collections import defaultdict

import json
import h5py
import pandas as pd
import re
import string
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import itertools
import gc

class TrimExceptAscii:
    def __call__(self, sentence):
        if isinstance(sentence, list):
            return sentence
        else:
            s = sentence.encode('ascii', 'ignore')
            return s
        
class Lowercase:
    def __call__(self, sentence):
        return sentence.lower()

class RemovePunctuation:
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, sentence):
        return self.regex.sub('', sentence.decode('ascii'))

class SplitWithWhiteSpace:

    def __call__(self, sentence):
        return sentence.split()

class Truncate:
    def __init__(self, n_word):
        self.n_word = n_word

    def __call__(self, words):
        return words[:self.n_word]
    

class MSVDDataset(Dataset):
    """ Dataset """

    def __init__(self, path, phase, tokenizer, max_frame=20, max_seq_len=20):
        self.data_path = path
        self.max_frame = max_frame
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.max_seq_len),
        ])
        self.tokenizer = tokenizer

        self.features = {}
        self.data = []
        self.load_data()
        gc.collect()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid, caption = self.data[idx]
        semantic_feature = self.features[vid]['semantic']
        semantic_mask = self.features[vid]['semantic_mask']

        visual_mask = self.features[vid]['visual_mask'] 
        node_feature = self.features[vid]['visual']     
        edge_index = torch.tensor(list(map(list, itertools.product(np.arange(edge_feature.shape[0]), repeat=2))), dtype=torch.long)
        edge_feature = self.features[vid]['edge'] 

            
        caption = ' '.join(caption)
        if self.tokenizer is None:
            input_ids = []
            attention_mask = []
        else: 
            tokenized = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_len + 1,
                return_tensors='pt',
                return_attention_mask=True,
                add_special_tokens=True
            )
            input_ids = tokenized.input_ids.squeeze(0)  
            attention_mask = tokenized.attention_mask.squeeze(0)
        input_ids = tokenized.input_ids.squeeze(0)  
        attention_mask = tokenized.attention_mask.squeeze(0)
        return vid, node_feature, edge_index, edge_feature, visual_mask,\
            semantic_feature, semantic_mask, input_ids, attention_mask
    
    def load_feat(self, video_ids):
        feature_list = ['visual', 'edge', 'semantic']
        for feature_name in feature_list:
            file_path = f'{self.data_path}/data/MSVD/features/MSVD_{feature_name}_clip.hdf5'
            if feature_name == "edge":
                with h5py.File(file_path, "r") as fs:
                    for key in video_ids:
                        if key not in self.features.keys():
                            self.features[key] = {}

                    feature = torch.from_numpy(fs[key][()])
                    self.features[key][feature_name] = feature
            else: 
                with h5py.File(file_path, "r") as fs:
                    for key in video_ids:
                        if key not in self.features.keys():
                            self.features[key] = {}

                        feature = torch.from_numpy(fs[key][()])
                        num_frames, feat_dim = feature.shape
                        pad_len = self.max_frame - num_frames
                        
                        if pad_len < 0:
                            feature = feature[:self.max_frame]
                            mask = torch.zeros(self.max_frame, dtype=torch.bool)
                        else:
                            pad_feat = torch.zeros((pad_len, feat_dim), dtype=feature.dtype)
                            feature = torch.cat([feature, pad_feat], dim=0)
                            mask = torch.cat([
                                torch.zeros(num_frames, dtype=torch.bool),
                                torch.ones(pad_len, dtype=torch.bool)
                            ])
                            
                        self.features[key][feature_name] = feature
                        self.features[key][f'{feature_name}_mask'] = mask

    def load_data(self):
        caption_fpath = f'{self.data_path}/data/MSVD/metadata/{self.phase}.csv'
        df = pd.read_csv(caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[[ 'VideoID', 'Start', 'End', 'Description' ]]
        df = df[pd.notnull(df['Description'])]

        captions = defaultdict(lambda: [])
        for video_id, start, end, caption in df.values:
            vid = "{}_{}_{}".format(video_id, start, end)
            captions[vid].append(caption)

        video_ids = []
        for vid in captions.keys():
            video_ids.append(vid)
            for caption in captions[vid]:
                self.data.append((vid, self.transform_sentence(caption)))

        self.load_feat(video_ids)

class MSRVTTDataset(Dataset):
    """ Dataset """

    def __init__(self, path, phase, tokenizer, max_frame=20, max_seq_len=20):
        self.data_path = path
        self.max_frame = max_frame
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.max_frame),
        ])
        self.tokenizer = tokenizer

        self.features = {}
        self.data = []
        self.load_data()
        gc.collect()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid, caption = self.data[idx]
        semantic_feature = self.features[vid]['semantic']
        semantic_mask = self.features[vid]['semantic_mask']

        visual_mask = self.features[vid]['visual_mask'] 
        node_feature = self.features[vid]['visual']     
        edge_index = torch.tensor(list(map(list, itertools.product(np.arange(edge_feature.shape[0]), repeat=2))), dtype=torch.long)
        edge_feature = self.features[vid]['edge'] 

            
        caption = ' '.join(caption)
        if self.tokenizer is None:
            input_ids = []
            attention_mask = []
        else: 
            tokenized = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_len + 1,
                return_tensors='pt',
                return_attention_mask=True,
                add_special_tokens=True
            )
            input_ids = tokenized.input_ids.squeeze(0)  
            attention_mask = tokenized.attention_mask.squeeze(0)
        input_ids = tokenized.input_ids.squeeze(0)  
        attention_mask = tokenized.attention_mask.squeeze(0)
        return vid, node_feature, edge_index, edge_feature, visual_mask,\
            semantic_feature, semantic_mask, input_ids, attention_mask

    def load_feat(self, video_ids):
        feature_list = ['visual', 'edge', 'semantic']
        for feature_name in feature_list:
            file_path = f'{self.data_path}/data/MSR-VTT/features/MSR-VTT_{feature_name}_clip.hdf5'
            if feature_name == "edge":
                with h5py.File(file_path, "r") as fs:
                    for key in video_ids:
                        if key not in self.features.keys():
                            self.features[key] = {}

                    feature = torch.from_numpy(fs[key][()])
                    self.features[key][feature_name] = feature
            else: 
                with h5py.File(file_path, "r") as fs:
                    for key in video_ids:
                        if key not in self.features.keys():
                            self.features[key] = {}

                        feature = torch.from_numpy(fs[key][()])
                        num_frames, feat_dim = feature.shape
                        pad_len = self.max_frame - num_frames
                        
                        if pad_len < 0:
                            feature = feature[:self.max_frame]
                            mask = torch.zeros(self.max_frame, dtype=torch.bool)
                        else:
                            pad_feat = torch.zeros((pad_len, feat_dim), dtype=feature.dtype)
                            feature = torch.cat([feature, pad_feat], dim=0)
                            mask = torch.cat([
                                torch.zeros(num_frames, dtype=torch.bool),
                                torch.ones(pad_len, dtype=torch.bool)
                            ])
                            
                        self.features[key][feature_name] = feature
                        self.features[key][f'{feature_name}_mask'] = mask

    def load_data(self):
        caption_fpath = f'{self.data_path}/data/MSR-VTT/metadata/{self.phase}.json'
        with open(caption_fpath, 'r') as fin:
            data = json.load(fin)

        captions = defaultdict(lambda: [])
        for vid, depth1 in data.items():
            for caption in depth1.values():
                captions[vid].append(caption)
        
        video_ids = []
        for vid in captions.keys():
            video_ids.append(vid)
            for caption in captions[vid]:
                self.data.append((vid, self.transform_sentence(caption)))

        self.load_feat(video_ids)