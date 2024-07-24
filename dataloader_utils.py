import logging
import torch
import pandas as pd
import json
import re
from torch.utils.data import DataLoader, Dataset
from functools import partial
from mpi4py import MPI
import os
import random
import numpy as np
from PIL import Image
import skimage.io as io1
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import pickle
logging.basicConfig(level=logging.INFO)

def get_dataloader(tokenizer, data_path, label_path, batch_size, max_seq_len, max_seq_len_src, args):

    dataset = TextDataset_translation(tokenizer=tokenizer, data_path=data_path,   #source=args.src, target=args.tgt,
                                        label_path=label_path)
                                        # shard=MPI.COMM_WORLD.Get_rank(),
                                        # num_shards=MPI.COMM_WORLD.Get_size())
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle='train' in data_path,
        num_workers=4,
        collate_fn=partial(TextDataset_translation.collate_pad, 
                           data_path=data_path,
                           cutoff=max_seq_len, 
                           cutoff_src=max_seq_len_src,
                           padding_token=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.get_vocab()['<pad>']),
    )

    while True:
        for batch in dataloader:
            yield batch

# class TextDataset(Dataset):
#     def __init__(
#         self,
#         tokenizer,
#         data_path: str,
#         has_labels: bool = False
#         ) -> None:
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.read_data()
#         if has_labels:
#             self.read_labels()

#     def read_data(self):
#         logging.info("Reading data from {}".format(self.data_path))
#         data = pd.read_csv(self.data_path, sep="\t", header=None)  # read text file
#         logging.info(f"Tokenizing {len(data)} sentences")

#         self.text = data[0].apply(lambda x: x.strip()).tolist()
#         if hasattr(self.tokenizer, 'encode_batch'):

#             encoded_input = self.tokenizer.encode_batch(self.text)
#             self.input_ids = [x.ids for x in encoded_input]
        
#         else:
#             encoded_input = self.tokenizer(self.text)
#             self.input_ids = encoded_input["input_ids"]

        

#     def read_labels(self):
#         self.labels = pd.read_csv(self.data_path, sep="\t", header=None)[1].tolist()
#         # check if labels are already numerical
#         self.labels = [str(x) for x in self.labels]
#         if isinstance(self.labels[0], int):
#             return
#         # if not, convert to numerical
#         all_labels = sorted(list(set(self.labels)))
#         self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
#         self.idx_to_label = {i: label for i, label in self.label_to_idx.items()}
#         self.labels = [self.label_to_idx[label] for label in self.labels]
        
        
    
#     def __len__(self) -> int:
#         return len(self.text)

#     def __getitem__(self, i):
#         out_dict = {
#             "input_ids": self.input_ids[i],
#             # "attention_mask": [1] * len(self.input_ids[i]),
#         }
#         if hasattr(self, "labels"):
#             out_dict["label"] = self.labels[i]
#         return out_dict

#     @staticmethod
#     def collate_pad(batch, cutoff: int):
#         max_token_len = 0
#         num_elems = len(batch)
#         # batch[0] -> __getitem__[0] --> returns a tuple (embeddings, out_dict)

#         for i in range(num_elems):
#             max_token_len = max(max_token_len, len(batch[i]["input_ids"]))

#         max_token_len = min(cutoff, max_token_len)

#         tokens = torch.zeros(num_elems, max_token_len).long()
#         tokens_mask = torch.zeros(num_elems, max_token_len).long()
        
#         has_labels = False
#         if "label" in batch[0]:
#             labels = torch.zeros(num_elems).long()
#             has_labels = True

#         for i in range(num_elems):
#             toks = batch[i]["input_ids"]
#             length = len(toks)
#             tokens[i, :length] = torch.LongTensor(toks)
#             tokens_mask[i, :length] = 1
#             if has_labels:
#                 labels[i] = batch[i]["label"]
        
#         # TODO: the first return None is just for backward compatibility -- can be removed
#         if has_labels:
#             return None, {"input_ids": tokens, "attention_mask": tokens_mask, "labels": labels}
#         else:
#             return None, {"input_ids": tokens, "attention_mask": tokens_mask}


# class BaseDataset(Dataset):
#     # def __init__(self, args, tokenizer, split, transform=None):
#     def __init__(self, tokenizer, data_path: str, shard, num_shards, max_seq_len) -> None:
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.shard = shard
#         self.num_shards = num_shards
#         self.max_seq_length = max_seq_length

#         if 'train' in self.data_path:
#             self.split = 'train'
#         elif 'val' in self.data_path:
#             self.split = 'val'
#         else:
#             self.split = 'test'
            
#         if self.split == 'train':
#             self.transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))])

#         if 'iu_xray' in self.data_path:
#             self.read_data_iu_xray()
#         else:
#             self.read_data_mimic_cxr()

#         self.image_dir = args.image_dir
#         self.ann_path = args.ann_path
#         self.max_seq_length = args.max_seq_length
#         self.split = split
#         self.tokenizer = tokenizer
#         self.transform = transform

#         # self.ann = json.loads(open(self.ann_path, 'r').read())
#         # self.examples = self.ann[self.split]
#         # with open(args.label_path, 'rb') as f:
#         #     self.labels = pickle.load(f)

#         # for i in range(len(self.examples)):
#         #     self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
#         #     self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])


class TextDataset_translation(Dataset):

    def __init__(
        self,
        tokenizer,
        data_path: str,
        label_path: str,        
        # shard,
        # num_shards,
        ) -> None:
        self.data_path = data_path
        self.tokenizer = tokenizer
        # self.shard = shard
        # self.num_shards = num_shards
        self.label_path = label_path

        if 'train' in self.data_path:
            self.split = 'train'
        elif 'val' in self.data_path:
            self.split = 'val'
        else:
            self.split = 'test'
            
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if 'iu_xray' in self.data_path:
            self.read_data_iu_xray()
        else:
            self.read_data_mimic_cxr()


    def read_data_iu_xray(self):
        
        # use two images stack
        dataset = json.loads(open(f"{self.data_path}/../annotation_v1.json", 'r').read())
        # dataset = json.loads(open(f"{self.data_path}/../annotation_labels.json", 'r').read())
        examples = dataset[self.split]

        with open(self.label_path, 'rb') as f:
            labels = pickle.load(f)

        tgt_text = []
        image_path = []
        image_id = []
        self.labels = []
        for i in range(len(examples)):
            txt = examples[i]['report']
            img = examples[i]['image_path']

            _id = examples[i]['id']
            array = _id.split('-')
            modified_id = array[0]+'-'+array[1]
            label = torch.FloatTensor(labels[modified_id])

            tgt_text.append(txt)
            image_path.append(img)
            image_id.append(_id)
            self.labels.append(label)

        self.tgt_text = tgt_text
        self.image_path = image_path
        self.image_id = image_id
        # if hasattr(self.tokenizer, 'encode_batch'):
        #     self.input_ids_src = []
        #     for x in self.image_path:
        #         image_1 = Image.open(os.path.join(f"{self.data_path}/../images", x[0])).convert('RGB')
        #         image_2 = Image.open(os.path.join(f"{self.data_path}/../images", x[1])).convert('RGB')
        #         if self.transform is not None:
        #             image_1 = self.transform(image_1)
        #             image_2 = self.transform(image_2)
        #         image = torch.stack((image_1, image_2), 0)
        #         self.input_ids_src.append(image)
            
        #     encoded_input_tgt = self.tokenizer.encode_batch(self.tgt_text)
        #     self.input_ids_tgt = [x.ids for x in encoded_input_tgt]
        # else:
        #     pass

        encoded_input_tgt = self.tokenizer.encode_batch(self.tgt_text)
        self.input_ids_tgt = [x.ids for x in encoded_input_tgt]

        count_length_tgt = np.mean([len(item) for item in self.input_ids_tgt])

        # print(f'average number of tokens in source {count_length_src}')
        print(f'average number of tokens in target {count_length_tgt}')
    
    def read_data_mimic_cxr(self):
        dataset = json.loads(open(f"{self.data_path}/../annotation_v1.json", 'r').read())
        examples = dataset[self.split]
        
        with open(self.label_path, 'rb') as f:
            labels = pickle.load(f)

        tgt_text = []
        image_path = []
        image_id = []
        self.labels = []
        for i in range(len(examples)):
            txt = examples[i]['report']
            img = examples[i]['image_path']
            _id = examples[i]['id']
            label = torch.FloatTensor(labels[_id])

            tgt_text.append(txt)
            image_path.append(img)
            image_id.append(_id)
            self.labels.append(label)

        self.tgt_text = tgt_text
        self.image_path = image_path
        self.image_id = image_id
        
        # if hasattr(self.tokenizer, 'encode_batch'):
        #     self.input_ids_src = []
        #     for x in self.image_path:
        #         image = Image.open(os.path.join(f"{self.data_path}/../images", x[0])).convert('RGB')
        #         if self.transform is not None:
        #             image = self.transform(image)
        #         self.input_ids_src.append(image)
                
        #     encoded_input_tgt = self.tokenizer.encode_batch(self.tgt_text)
        #     self.input_ids_tgt = [x.ids for x in encoded_input_tgt]

        # else:
        #     pass
        encoded_input_tgt = self.tokenizer.encode_batch(self.tgt_text)
        self.input_ids_tgt = [x.ids for x in encoded_input_tgt]

        count_length_tgt = np.mean([len(item) for item in self.input_ids_tgt])

        # print(f'average number of tokens in source {count_length_src}')
        print(f'average number of tokens in target {count_length_tgt}')

    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, i):
        x = self.image_path[i]
        if 'iu_xray' in self.data_path:
            image_1 = Image.open(os.path.join(f"{self.data_path}/../images", x[0])).convert('RGB')
            image_2 = Image.open(os.path.join(f"{self.data_path}/../images", x[1])).convert('RGB')
            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)
        else:
            image = Image.open(os.path.join(f"{self.data_path}/../images", x[0])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

        # encoded_input_tgt = self.tokenizer.encode_batch(self.tgt_text[i])
        # self.input_ids_tgt = [x.ids for x in encoded_input_tgt]


        
        out_dict = {
            # "encoder_input_ids": self.input_ids_src[i],
            "encoder_input_ids": image,
            "decoder_input_ids": self.input_ids_tgt[i],
            "image_id": self.image_id[i],
            "label": self.labels[i]

        }
        return out_dict

    @staticmethod
    def collate_pad(batch, data_path, cutoff: int, cutoff_src: int, padding_token: int):
        max_token_len_src, max_token_len_tgt = cutoff_src, cutoff
        num_elems = len(batch)

        tokens_tgt = torch.ones(num_elems, max_token_len_tgt).long() * padding_token
        if 'iu_xray' in data_path:
            tokens_src = torch.ones(num_elems, 2, 3, 224, 224)
        else:
            tokens_src = torch.ones(num_elems, 3, 224, 224)
        tokens_mask_tgt = torch.zeros(num_elems, max_token_len_tgt).long()
        image_ids = []

        tokens_label = torch.FloatTensor(num_elems, 14)

        for i in range(num_elems):
            image = batch[i]['encoder_input_ids']
            image_id = batch[i]['image_id']
            image_ids.append(image_id)

            label = batch[i]['label']
            tokens_label[i] = label


            tokens_src[i] = image
            toks_tgt = batch[i]["decoder_input_ids"][:max_token_len_tgt]
            l_t = len(toks_tgt)

            tokens_tgt[i, :l_t] = torch.LongTensor(toks_tgt)
            tokens_mask_tgt[i, :] = 1

        return {'input_embed': tokens_src, 'input_ids': None, 'image_id': image_ids, 'label': tokens_label, #"input_ids": tokens_src, "attention_mask": tokens_mask_src, 
                    'decoder_input_ids': tokens_tgt, 'decoder_attention_mask': tokens_mask_tgt}, None
