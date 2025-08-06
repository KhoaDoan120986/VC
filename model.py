from modules.module_visual import NormalizeVideo, VisualModel, VisualConfig
from modules.until_module import LayerNorm
from transformers.modeling_outputs import BaseModelOutput
from torch_geometric.nn import TransformerConv
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
import torch.nn as nn
import torch 


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output  = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class TransC(nn.Module):
    def __init__(self, node_feat_dim, d_model, edge_dim, heads=4, project_edge_dim=None, more_skip=True, last_average=False, beta=True):
        super().__init__()
        self.lp = nn.Linear(node_feat_dim, d_model)
        self.more_skip = more_skip
        self.project_edge_dim = project_edge_dim
        if self.project_edge_dim is not None:
            self.lp_edge_attr = nn.Linear(edge_dim, project_edge_dim)
            edge_dim = project_edge_dim
        
        self.conv1 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)
        
        self.conv2 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)
        
        if last_average:
            self.conv3 = TransformerConv(d_model, d_model, heads, concat=False, edge_dim=edge_dim, aggr='mean', beta=beta)
        else:
            self.conv3 = TransformerConv(d_model, int(d_model/heads), heads, edge_dim=edge_dim, aggr='mean', beta=beta)

    def forward(self, data):
        x = self.lp(data.x)
        if self.project_edge_dim is not None:
            e = F.relu(self.lp_edge_attr(data.edge_attr))
        else:
            e = data.edge_attr
        if self.more_skip:
            x = F.relu(x + self.conv1(x, data.edge_index, e))
            x = F.relu(x + self.conv2(x, data.edge_index, e))
            x = F.relu(x + self.conv3(x, data.edge_index, e))
        else:
            x = F.relu(self.conv1(x, data.edge_index, e))
            x = F.relu(self.conv2(x, data.edge_index, e))
            x = F.relu(self.conv3(x, data.edge_index, e))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, task_config, type_vocab_size=2):
        super(Encoder, self).__init__()
        self.task_config = task_config
        
        self.visual_config, _ = VisualConfig.get_config(
            'visual-base', None, type_vocab_size,
            state_dict=None, task_config=self.task_config)

        assert self.task_config.max_frames <= self.visual_config.max_position_embeddings
        self.visual_config.num_hidden_layers = self.task_config.visual_num_hidden_layers

        # ====> Graph <====
        self.transc = TransC(node_feat_dim=task_config.node_feat_dim, 
                            d_model=task_config.d_graph, edge_dim=task_config.edge_dim, 
                            project_edge_dim=task_config.project_edge_dim,
                            more_skip=task_config.no_skip==False, last_average=task_config.last_average,
                            beta=task_config.no_beta_transformer==False)
        
        self.visual = VisualModel(self.visual_config)
        self.visual_normalize_video = NormalizeVideo(self.visual_config.vocab_size)
    
        # ====> Semantic <====
        self.semantic_embeddings = nn.Sequential(
            LayerNorm(self.task_config.semantic_dim),
            nn.Linear(self.task_config.semantic_dim, self.visual_config.hidden_size)
        )

        self.semantic = nn.ModuleList([
            EncoderLayer(self.visual_config.hidden_size, self.visual_config.num_attention_heads)
            for _ in range(self.task_config.visual_num_hidden_layers)
        ])
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights. """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.visual_config.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'beta') and hasattr(module, 'gamma'):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, geo_graph, video_mask, kg, kg_mask, shaped=False):
        # ====> Graph <====
        batch = video_mask.shape[0]
        n_nodes = geo_graph.x.shape[0] // batch
        fo_convolved = self.transc(geo_graph)
        fo_convolved = fo_convolved.unflatten(0, (batch, n_nodes))

        if shaped is False:
            video = self.visual_normalize_video(fo_convolved)

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]

        # ====> Semantic <====
        kg_output = self.semantic_embeddings(kg)
        for i in range(self.task_config.visual_num_hidden_layers):
            kg_output = self.semantic[i](kg_output, kg_mask)

        return visual_output, kg_output
        
class Decoder(nn.Module):
    def __init__(self, tokenizer, task_config, model_class):
        super(Decoder, self).__init__()
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.model = model_class.from_pretrained(
            task_config.decoder_name,
            torch_dtype=torch.float32,
            device_map=None
        )

        if hasattr(self.model, "vision_model"):
            for p in self.model.vision_model.parameters():
                p.requires_grad = False

        for p in self.model.qformer.parameters():
            p.requires_grad = False
        for p in self.model.language_model.parameters():
            p.requires_grad = True

        self.model.language_model.decoder.block = self.model.language_model.decoder.block[:self.task_config.visual_num_hidden_layers]
        self.vocab_size = self.model.language_model.config.vocab_size
        
        # ===> ÁP DỤNG LoRA <===
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "k", "v", "o"],  # attention projections trong T5
            lora_dropout=0.01,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        
    def forward(self, encoder_hidden_states, video_mask, caption, caption_mask):
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        outputs = self.model.language_model(
            encoder_outputs=encoder_outputs,
            attention_mask=video_mask,            # mask cho encoder
            decoder_input_ids=caption,            # input captions (shifted)
            decoder_attention_mask=caption_mask,  # mask cho decoder
            return_dict=True,
        )
        return outputs
    
    def generate(self, encoder_hidden_states, video_mask):
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        generated = self.model.language_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=video_mask,
            decoder_start_token_id=self.tokenizer.eos_token_id,
            max_length=self.task_config.max_seq_len + 1,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return generated
    
class VCModel(nn.Module):
    def __init__(self, tokenizer, task_config, decoder_model_class):
        super().__init__()
        self.task_config = task_config
        self.encoder = Encoder(task_config)
        self.decoder = Decoder(tokenizer, task_config, decoder_model_class)
        self.visual_lp = nn.Linear(self.task_config.hidden_size, self.decoder.model.language_model.config.d_model)
        self.semantic_lp = nn.Linear(self.task_config.hidden_size, self.decoder.model.language_model.config.d_model)

    def forward(self, geo_graph, video_mask, kg, kg_mask, caption, caption_mask):
        visual_output, kg_output = self.encoder(geo_graph, video_mask, kg, kg_mask)
        visual_output = self.visual_lp(visual_output)
        kg_output = self.semantic_lp(kg_output)
        encoder_hidden_states = visual_output + kg_output
        decoder_output = self.decoder(encoder_hidden_states, video_mask, caption, caption_mask)
        return decoder_output
    
    def generate(self, geo_graph, video_mask, kg, kg_mask):
        visual_output, kg_output = self.encoder(geo_graph, video_mask, kg, kg_mask)
        visual_output = self.visual_lp(visual_output)
        kg_output = self.semantic_lp(kg_output)
        encoder_hidden_states = visual_output + kg_output
        generated = self.decoder.generate(encoder_hidden_states, video_mask)
        return generated