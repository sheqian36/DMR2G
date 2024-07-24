import json
import numpy as np

import pandas as pd

# from metrics import compute_scores
import json
import re
from collections import Counter


from bleu.bleu import Bleu
from meteor import Meteor
from rouge import Rouge
from cider.cider import Cider

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDER")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res




import logging
# dataset_name = "iu_xray"
# dataset_name = "mimic_cxr"
# create tokenizer
# tokenizer = Tokenizer(dataset_name)
input_file = "/home/shuchenweng/cz/oyh/output/seqdiffuseq/mimic_cxr_ckpts/debug/ema_0.9999_140000.pt.samples_3858.steps-2000.clamp-no_clamp-normal_10708.txt"
test_res = []
test_gts = []
with open(input_file, 'r') as f:
    data = f.readlines()
    data = [json.loads(item.strip('\n')) for item in data]


logging.basicConfig(filename=f"debug_mimic_cxr_log.txt", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

for i in data:
    res = i[0]
    gts = i[1]
    
    test_res.append(res)
    test_gts.append(gts)


log = dict()
test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                            {i: [re] for i, re in enumerate(test_res)})
log.update(**{'test_' + k: v for k, v in test_met.items()})

# 将日志写入文件
# step = input_file.split('/')[-1].split('_')[2]
# logging.info(f'{step}')
for key, value in log.items():
    logging.info(f'{key}: {value}')

print(log)

# test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
# test_res.to_csv(f"{name}_{step}_res.csv", index=False, header=False)
# test_gts.to_csv(f"{name}_{step}_gts.csv", index=False, header=False)


# path = '/home/u2020211076/jupyterlab/SeqDiffuSeq-main/ckpts/iu_xray_ckpts/Medclip_BartEncoder_label_200000/ema_0.9999_160000.pt_res.json'
# res = json.loads(open(path, 'r').read())

# path2 = '/home/u2020211076/jupyterlab/SeqDiffuSeq-main/ckpts/iu_xray_ckpts/Medclip_BartEncoder_label_200000/ema_0.9999_160000.pt_gts.json'
# gts = json.loads(open(path2, 'r').read())

# e = []
# test_gts, test_res = [], []
# for key, value in res.items():
#     report = value.split('cicatrix  lesion  ')[1]
#     # ids = tokenizer(value)[:60]
#     # reports = tokenizer.decode(ids[1:])
#     test_res.append(report)

# f = []
# for key, value in gts.items():
#     # print(value)
#     report = value.split('cicatrix  lesion  ')[1]
#     # ids = tokenizer(value)[:60]
#     # gts = tokenizer.decode(ids[1:])
#     test_gts.append(report)

# import logging

# # 配置日志记录器
# name = path.split('/')[-2]
# logging.basicConfig(filename=f"{name}_log.txt", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log = dict()
# test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
#                             {i: [re] for i, re in enumerate(test_res)})
# log.update(**{'test_' + k: v for k, v in test_met.items()})

# # 将日志写入文件
# step = path.split('/')[-1].split('_')[2]
# logging.info(f'{step}')
# for key, value in log.items():
#     logging.info(f'{key}: {value}')

# print(log)
    
# test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
# test_res.to_csv("res.csv", index=False, header=False)
# test_gts.to_csv("gts.csv", index=False, header=False)

    # # create data loader
    # train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    # val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    # test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # # build model architecture
    # model = R2GenModel(args, tokenizer)

    # # get function handles of loss and metrics
    # criterion = compute_loss
    # metrics = compute_scores

    # # build optimizer, learning rate scheduler
    # optimizer = build_optimizer(args, model)
    # lr_scheduler = build_lr_scheduler(args, optimizer)

    # # build trainer and start to train
    # trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    # trainer.train()


# if __name__ == '__main__':
#     main()
