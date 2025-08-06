from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [[] for _ in range(self.num_losses)]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [0. for _ in range(self.num_losses)]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch, device):
    vid, node_feature, edge_index, edge_feature, visual_mask,\
            semantic_feature, semantic_mask, input_ids, attention_mask = batch

    node_feature = node_feature.to(device)
    edge_index = edge_index.to(device)
    edge_feature = edge_feature.to(device)
    visual_mask = visual_mask.to(device)

    semantic_feature = semantic_feature.to(device)
    semantic_mask = semantic_mask.to(device)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    return vid, node_feature, edge_index, edge_feature, visual_mask,\
            semantic_feature, semantic_mask, input_ids, attention_mask

def train(epoch, data_loader, model, tokenizer, gradient_accumulation_steps, optimizer, lr_scheduler, gradient_clip, device):
    model.train()
    loss_checker = LossChecker(2)
    vocab_size = model.decoder.vocab_size
    t = tqdm(data_loader)
    for step, batch in enumerate(t):
        vid, geo_x, geo_edge_index, geo_edge_attr, visual_mask,\
            semantic_feature, semantic_mask, captions, attention_mask = parse_batch(batch, device)

        input_ids = captions[:, :-1]
        labels = captions[:, 1:].clone()
        input_mask = attention_mask[:, :-1]
        
        batch_sz = geo_x.shape[0]
        x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])
        offset = []
        for i in range(batch_sz):
            n_edges = geo_edge_index[0].shape[1]
            offset_val = int(np.sqrt(n_edges)) * i
            offset.append(torch.full(geo_edge_index[0].shape, offset_val))
        offset = torch.stack(offset).cuda()
        geo_graph_batch_offset = geo_edge_index + offset

        new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
        edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)
        edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                geo_edge_attr.shape[2]).float()

        data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
        
        output = model(data_geo_graph_batch, visual_mask,
                       semantic_feature, semantic_mask,
                       caption=input_ids, caption_mask=input_mask, labels=labels)

        loss = output.loss
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()
        loss_checker.update(loss.item(), 0.0)
        t.set_description("[Epoch #{0}] loss: {1:.3f}, {2:.3f}".format(epoch, *loss_checker.mean(last=10)))

    total_loss,_ = loss_checker.mean()
    return total_loss

def test(epoch, data_loader, model, tokenizer, device):
    model.eval()
    loss_checker = LossChecker(num_losses=2)
    vocab_size = model.decoder.vocab_size
    t = tqdm(data_loader, desc='Test: ')
    with torch.no_grad():
        for step, batch in enumerate(t):
            vid, geo_x, geo_edge_index, geo_edge_attr, visual_mask,\
            semantic_feature, semantic_mask, captions, attention_mask = parse_batch(batch, device)

            input_ids = captions[:, :-1]
            labels = captions[:, 1:].clone()
            input_mask = attention_mask[:, :-1]
            
            batch_sz = geo_x.shape[0]
            x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])
            offset = []
            for i in range(batch_sz):
                n_edges = geo_edge_index[0].shape[1]
                offset_val = int(np.sqrt(n_edges)) * i
                offset.append(torch.full(geo_edge_index[0].shape, offset_val))
            offset = torch.stack(offset).cuda()
            geo_graph_batch_offset = geo_edge_index + offset

            new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
            edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)
            edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                    geo_edge_attr.shape[2]).float()

            data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
        
            output = model(data_geo_graph_batch, visual_mask,
                       semantic_feature, semantic_mask,
                       caption=input_ids, caption_mask=input_mask, labels=labels)

            loss = output.loss
            loss_checker.update(loss.item(), 0.0)
            t.set_description("[Epoch #{0}] loss: {1:.3f}, {2:.3f}".format(epoch, *loss_checker.mean(last=10)))
    total_loss,_ = loss_checker.mean()
    return total_loss
    

def get_groundtruth_captions(tokenizer, data_loader):
    vid2GTs = {}
    for batch in tqdm(data_loader, desc='get_groundtruth_captions: '):
        video_ids = batch[0]
        captions = batch[-2]
        for i, vid in enumerate(video_ids):
            if vid not in vid2GTs:
                vid2GTs[vid] = []
            caption = tokenizer.decode(captions[i], skip_special_tokens=True)
            vid2GTs[vid].append(caption)
    return vid2GTs

def get_predicted_captions(model, tokenizer, data_loader, device):
    model.eval()
    vid2pred = {}
    videos = set()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='get_predicted_captions: '):
            video_ids, geo_x, geo_edge_index, geo_edge_attr, visual_mask,\
            semantic_feature, semantic_mask = batch[:7]
            for i, vid in enumerate(video_ids):
                if vid in videos:
                    continue
                videos.add(vid)
                data_geo_graph = Data(x=geo_x[i].to(device), 
                                    edge_index=geo_edge_index[i].to(device),
                                    edge_attr=geo_edge_attr[i].to(device))
                v_mask = visual_mask[i].unsqueeze(0).to(device)
                k_feat = semantic_feature[i].unsqueeze(0).to(device)
                k_mask = semantic_mask[i].unsqueeze(0).to(device)
                tokens = model.generate(data_geo_graph, v_mask, k_feat, k_mask)
                vid2pred[vid] = tokenizer.decode(tokens[0], skip_special_tokens=True)
                
    return vid2pred

def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = float(score)
    return final_scores

def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = {v: i for i, v in enumerate(vid2pred.keys())}
    refs = {vid2idx[vid]: GTs for vid, GTs in vid2GTs.items()}
    hypos = {vid2idx[vid]: [pred] for vid, pred in vid2pred.items()}

    scores = calc_scores(refs, hypos)
    return scores