import torch
import numpy as np
from .tokenizer import SimpleTokenizer as _Tokenizer

from copy import deepcopy
import torch.nn as nn

_tokenizer = _Tokenizer()

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class PromptPool(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 

        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.classnames = classnames

        self.prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
        self.prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        
        self.normal_num = design_details["num_normal"]+1
        self.anormaly_num = design_details["num_anormaly"]+1

        self.state_normal_list = ["{}" for i in range(self.normal_num)]
        self.state_anomaly_list = ["damaged {}" for i in range(self.anormaly_num)]

        # Random Initialization
        if True:
            print("Initializing class-specific contexts")
            ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
            ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
            ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
        prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(self.normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)
       
        self.vis_ctx = None
        if design_details["VPrompt_length"] != 0:
            self.VPT_shallow = True
            n_vis_ctx = design_details["VPrompt_length"]
            vis_ctx_vectors = torch.empty(n_vis_ctx, clip_model.visual.embed_dim, dtype=dtype)
            nn.init.normal_(vis_ctx_vectors, std=0.02)
            self.vis_ctx = nn.Parameter(vis_ctx_vectors)  # to be optimized
            self.register_buffer("vision_prompt", vis_ctx_vectors)
            print("vision_prompt shape", self.vis_ctx.shape)

    def get_original_text_tokens(self, cls_name):
        text = 'a photo of a '
        self.ori_num = 3
        original_text_tokens = []
        for i in range(len(cls_name)):
            ori_nomal_list = np.random.choice(self.prompt_normal, self.ori_num)
            ori_anomaly_list = np.random.choice(self.prompt_abnormal, self.ori_num)
            sample_nomal = []
            sample_anomaly = []
            for j in range(self.ori_num):
                ori_nomal_prompt = text+ori_nomal_list[j].format(cls_name[i])
                ori_nomal_tokens = tokenize(ori_nomal_prompt)
                sample_nomal.append(ori_nomal_tokens)
                ori_anomaly_prompt = text+ori_anomaly_list[j].format(cls_name[i])
                ori_anomaly_tokens = tokenize(ori_anomaly_prompt)
                sample_anomaly.append(ori_anomaly_tokens)
            original_text_tokens.append(torch.stack(sample_nomal+sample_anomaly, dim=1).squeeze(0))

        return torch.stack(original_text_tokens)

    def forward(self, cls_id =None):
        
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)

        return prompts, tokenized_prompts, self.compound_prompts_text, self.vis_ctx