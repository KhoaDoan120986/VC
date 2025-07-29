from modules.module_visual import NormalizeVideo, VisualModel, VisualConfig
from modules.until_module import LayerNorm
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
import torch 
from peft import LoraConfig, get_peft_model

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

    
class Encoder(nn.Module):
    def __init__(self, task_config, type_vocab_size=2):
        super(Encoder, self).__init__()
        self.task_config = task_config
        
        self.visual_config, _ = VisualConfig.get_config(
            'visual-base', None, type_vocab_size,
            state_dict=None, task_config=self.task_config)

        assert self.task_config.max_frames <= self.visual_config.max_position_embeddings
        self.visual_config.num_hidden_layers = self.task_config.visual_num_hidden_layers

        # ====> Visual <====
        self.visual_lp1 = nn.Linear(self.task_config.visual_dim, self.visual_config.vocab_size)
        self.visual = VisualModel(self.visual_config)
        self.visual_normalize_video = NormalizeVideo(self.visual_config.vocab_size)
        self.visual_lp2 = nn.Linear(self.visual_config.hidden_size, self.task_config.d_model)

        # ====> Motion <====
        self.motion_lp1 = nn.Linear(self.task_config.motion_dim, self.visual_config.vocab_size)
        self.motion = VisualModel(self.visual_config)
        self.motion_normalize_video = NormalizeVideo(self.visual_config.vocab_size)
        self.motion_lp2 = nn.Linear(self.visual_config.hidden_size, self.task_config.d_model)

        # ====> Semantic <====
        self.semantic_embeddings = nn.Sequential(
            LayerNorm(self.task_config.semantic_dim),
            nn.Linear(self.task_config.semantic_dim, self.visual_config.hidden_size)
        )

        self.semantic = nn.ModuleList([
            EncoderLayer(self.visual_config.hidden_size, self.visual_config.num_attention_heads)
            for _ in range(self.task_config.visual_num_hidden_layers)
        ])
        self.semantic_lp = nn.Linear(self.visual_config.hidden_size, self.task_config.d_model)

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

    def forward(self, video, video_mask, motion, motion_mask, kg, kg_mask, shaped=False):
        assert video.shape[2] == self.task_config.visual_dim
        assert motion.shape[2] == self.task_config.motion_dim
        assert kg.shape[2] == self.task_config.semantic_dim

        # ====> Visual <====
        video = self.visual_lp1(video)
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.visual_normalize_video(video)

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        visual_output = self.visual_lp2(visual_output)

        # ====> Motion <====
        motion = self.motion_lp1(motion)
        if shaped is False:
            motion_mask = motion_mask.view(-1, motion_mask.shape[-1])
            motion = self.motion_normalize_video(motion)

        motion_layers, _ = self.motion(motion, motion_mask, output_all_encoded_layers=True)
        motion_output = motion_layers[-1]
        motion_output = self.motion_lp2(motion_output)

        # ====> Semantic <====
        kg_output = self.semantic_embeddings(kg)
        for i in range(self.task_config.visual_num_hidden_layers):
            kg_output = self.semantic[i](kg_output, kg_mask)
        kg_output = self.semantic_lp(kg_output)

        return visual_output + motion_output + kg_output
        
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

        # Unfreeze Q-Former và Decoder
        for p in self.model.qformer.parameters():
            p.requires_grad = True
        for p in self.model.language_model.parameters():
            p.requires_grad = True

        self.model.language_model.decoder.block = self.model.language_model.decoder.block[:4]
        self.vocab_size = self.model.language_model.config.vocab_size

        # ===> ÁP DỤNG LoRA <===
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "k", "v", "o"],  # attention projections trong T5
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)

    def forward(self, encoder_hidden_states, caption=None, caption_mask=None):
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        if caption is not None:
            outputs = self.model.language_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=caption,           
                attention_mask=caption_mask,
                return_dict=True,
            )
            return outputs
    def generate(self, encoder_hidden_states):
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        generated = self.model.language_model.generate(
            encoder_outputs=encoder_outputs,
            max_length=self.task_config.max_seq_len + 1,
            num_beams=5,
            early_stopping=True,
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
    def forward(self, video, video_mask, motion, motion_mask, kg, kg_mask, caption=None, caption_mask=None):
        encoder_hidden_states = self.encoder(video, video_mask, motion, motion_mask, kg, kg_mask)
        decoder_output = self.decoder(encoder_hidden_states, caption, caption_mask)
        return decoder_output
    def generate(self, video, video_mask, motion, motion_mask, kg, kg_mask):
        encoder_hidden_states = self.encoder(video, video_mask, motion, motion_mask, kg, kg_mask)
        generated = self.decoder.generate(encoder_hidden_states)
        return generated