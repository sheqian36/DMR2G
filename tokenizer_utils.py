import json
import logging
import pathlib
import torch
from transformers import AutoTokenizer
import re

from tokenizers.processors import BertProcessing
from tokenizers.implementations  import ByteLevelBPETokenizer
from tokenizers import decoders
logging.basicConfig(level=logging.INFO)

def create_tokenizer(return_pretokenized, path, tokenizer_type: str = "word-level", tokenizer_ckpt: str = None):
    
    if return_pretokenized:
        print(f'*******use pretrained tokenizer*****{return_pretokenized}*******')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        return tokenizer

    if tokenizer_type == "byte-level":
        return read_byte_level(path)
    elif tokenizer_type == "word-level":
        return read_word_level(path)
    else:
        raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")

def train_bytelevel(
    path, #list
    save_path,
    vocab_size=10000,
    min_frequency=10,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
):

    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(
        files=path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.save_model(str(pathlib.Path(save_path)))

def read_byte_level(path: str):
    tokenizer = ByteLevelBPETokenizer(
        f"{path}/vocab.json",
        f"{path}/merges.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_truncation(max_length=512)

    with open(f"{path}/vocab.json", "r") as fin:
        vocab = json.load(fin)

    # add length method to tokenizer object
    tokenizer.vocab_size = len(vocab)

    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    tokenizer.decoder = decoders.ByteLevel()
    print(tokenizer.vocab_size)

    # print("<0>: ", tokenizer.encode("<0>").ids)

    # print(
    #     tokenizer.encode(
    #         "normal<0>cardiomegaly<0>scoliosis <0> fractures <0> effusion <0> thickening <0> pneumothorax <0> emphysema <0> pneumonia <0> edema <0> hernia <1> calcinosis <0> medical device <0> hypoinflation <0> airspace disease <0> atelectasis <0> opacity <0> cicatrix <0> lesion <0> the heart size and pulmonary vascularity appear within normal limits . a large hiatal hernia is noted . the lungs are free of focal airspace disease . no pneumothorax or pleural effusion is seen . degenerative changes are present in the spine ."
    #     ).ids
    # )

    # print(
    #     tokenizer.decode(
    #         tokenizer.encode(
    #             "normal <0> cardiomegaly <0> scoliosis <0> fractures <0> effusion <0> thickening <0> pneumothorax <0> emphysema <0> pneumonia <0> edema <0> hernia <1> calcinosis <0> medical device <0> hypoinflation <0> airspace disease <0> atelectasis <0> opacity <0> cicatrix <0> lesion <0> the heart size and pulmonary vascularity appear within normal limits . a large hiatal hernia is noted . the lungs are free of focal airspace disease . no pneumothorax or pleural effusion is seen . degenerative changes are present in the spine ."
    #         ).ids,
    #         skip_special_tokens=True,
    #     )
    # )

    # ids = tokenizer.encode(
    #     "normal<0>cardiomegaly<0> scoliosis <0> fractures <0> effusion <0> thickening <0> pneumothorax <0> emphysema <0> pneumonia <0> edema <0> hernia <1> calcinosis <0> medical device <0> hypoinflation <0> airspace disease <0> atelectasis <0> opacity <0> cicatrix <0> lesion <0> the heart size and pulmonary vascularity appear within normal limits . a large hiatal hernia is noted . the lungs are free of focal airspace disease . no pneumothorax or pleural effusion is seen . degenerative changes are present in the spine ."
    # ).ids
    # print(ids)
    # tensor = torch.tensor(ids)
    # print(tokenizer.decode(tensor.tolist(), skip_special_tokens=True))
    # print(f"Vocab size: {tokenizer.vocab_size}")

    return tokenizer


def read_word_level(path: str):

    from transformers import PreTrainedTokenizerFast

    logging.info(f"Loading tokenizer from {path}/word-level-vocab.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{str(pathlib.Path(path))}/word-level-vocab.json",
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        padding_side="right",
    )

    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    return tokenizer


def train_word_level_tokenizer(
    path: str,
    vocab_size: int = 10000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
):

    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import NFD, Lowercase, StripAccents
    from tokenizers.pre_tokenizers import Digits, Whitespace
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import WordLevelTrainer

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [Digits(individual_digits=True), Whitespace()]
    )
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )

    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files=[path], trainer=trainer)

    tokenizer.__len__ = property(lambda self: self.vocab_size)

    tokenizer.enable_truncation(max_length=512)

    print(tokenizer.encode("the red.").ids)

    print(tokenizer.encode("the red."))

    tokenizer.save(f"{str(pathlib.Path(path).parent)}/word-level-vocab.json")


def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def write_gts(path):

    if "iu_xray" in path:
        clean_report = clean_report_iu_xray
    else:
        clean_report = clean_report_mimic_cxr

    ann = json.loads(open(f"{path}annotation.json", 'r').read())
    p = f"{path}all.txt"
    with open(p, "w") as f:
        for i in ['train', 'val', 'test']:
            for j in ann[i]:
                example = clean_report(j['report'])
                f.write(example + "\n")


if __name__ == "__main__":
    import sys
    import os

    if sys.argv[1] == "train-word-level":
        train_word_level_tokenizer(path=sys.argv[2])
    elif sys.argv[1] == "train-byte-level":
        path = f"/home/shuchenweng/cz/oyh/data/seqdiffuseq/{sys.argv[2]}/"
        # write_gts(path)
        data_path = [path + item for item in os.listdir(path) if 'all' in item]
        train_bytelevel(path=data_path, vocab_size=int(sys.argv[3])+5, save_path=path)
    elif sys.argv[1] == "create":
        create_tokenizer(path=sys.argv[2])
