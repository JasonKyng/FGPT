import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse

from tqdm import tqdm
import numpy as np

import CustomCLIP.clip as clip
from CustomCLIP.prompt_learner import PromptPool
from utils.utils import setup_seed, get_logger, get_transform, get_similarity_map
from utils.dataset import Dataset
from utils.entropy import Entropy
from utils.loss import OrthogonalLoss, FocalLoss, BinaryDiceLoss
from utils.eot import OptimalTransport

def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CustomCLIPArgs = {"Prompt_length": args.n_ctx, 
                      "learnabel_text_embedding_depth": args.depth, 
                      "learnabel_text_embedding_length": args.t_n_ctx,
                      "VPrompt_length": args.n_vis_ctx,
                      "num_normal": args.num_n,
                      "num_anormaly": args.num_a,}

    model, _ = clip.load("ViT-L/14@336px", device=device, design_details = CustomCLIPArgs)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    ##########################################################################################
    prompt_pool = PromptPool(model.to("cpu"), CustomCLIPArgs)
    prompt_pool.to(device)
    model.to(device)
    model.visual.VVAttention(20)

    for _, param in model.named_parameters():
        param.requires_grad_(False)
    ##########################################################################################
    optimizer = torch.optim.AdamW(list(prompt_pool.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    
    # losses
    loss_ortho = OrthogonalLoss()
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    
    patch_size = 14
    img_entropy = Entropy(patch_size, args.image_size, args.image_size).to(device)
    model.eval()
    prompt_pool.train()

    pos_fre = torch.tensor([1.0 for i in range(args.num_n)]).to(device)
    neg_fre = torch.tensor([1.0 for i in range(args.num_a)]).to(device)
    for epoch in tqdm(range(args.epoch)):
        prompt_pool.train()
        loss_list = []
        image_loss_list = []
        loss_scl_text_list = []
        ortho_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label =  items['anomaly']
            cls_name = items['cls_name']
            gt = items['img_mask'].squeeze().to(device)
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            img_entropy_map = img_entropy(image)
            prompts, tokenized_prompts, compound_prompts_text, vis_prompts = prompt_pool(cls_id = None)
            text_features_ori = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            if vis_prompts is not None:
                image_features, patch_features = model.encode_image_learn(image, vis_prompts, args.features_list, vv_layer = 20)
            else:
                image_features, patch_features = model.encode_image(image, args.features_list, vv_layer = 20)
            # image_features = add_noise(image_features, noise_level=1)
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features_max_ = [text_features_ori[:prompt_pool.normal_num, :], 
                                  text_features_ori[prompt_pool.normal_num:, :] ]
            
            # Orth loss
            ortho_loss = loss_ortho(text_features_max_[0][1:])+loss_ortho(text_features_max_[1][1:])
            ortho_loss_list.append(ortho_loss.item())

            text_features_max = text_features
            text_probs_ = image_features_norm @ text_features_max.T

            pos_text_probs_ = text_probs_[:,1:prompt_pool.normal_num]*(1/pos_fre)
            neg_text_probs_ = text_probs_[:,prompt_pool.normal_num+1:]*(1/neg_fre)
            pos_text_weights_ = text_probs_[:,:prompt_pool.normal_num]
            neg_text_weights_ = text_probs_[:,prompt_pool.normal_num:]

            pos_topk_values, pos_topk_indices = torch.topk(pos_text_probs_, args.num_n, dim = 1)
            neg_topk_values, neg_topk_indices = torch.topk(neg_text_probs_, args.num_a, dim = 1)
            pos_mean_topk_values = torch.mean(pos_topk_values, dim = 1)
            neg_mean_topk_values = torch.mean(neg_topk_values, dim = 1)

            # Dynamic Prompt Selection 
            samples_text_features = []
            text_features_topk_ = []
            prob_weights_ = []
            loss_scl_text = 0
            for i in range(text_probs_.shape[0]):
                if pos_mean_topk_values[i] > neg_mean_topk_values[i]:
                    pos_idx = pos_topk_indices[i][:args.top_k]
                    neg_idx = neg_topk_indices[i][-args.top_k:]
                    pos = True
                else:
                    pos_idx = pos_topk_indices[i][-args.top_k:]
                    neg_idx = neg_topk_indices[i][:args.top_k]
                    pos = False

                pos_fre[pos_idx] += 1
                neg_fre[neg_idx] += 1

                text_pos_idx = torch.cat((torch.tensor([0]).to(device), pos_idx + 1))
                text_neg_idx = torch.cat((torch.tensor([0]).to(device), neg_idx + 1))
                text_pos_weights = pos_text_weights_[i][text_pos_idx].softmax(dim = 0)
                text_neg_weights = neg_text_weights_[i][text_neg_idx].softmax(dim = 0)

                # weighted pooling or average pooling
                text_features_max_pos = text_pos_weights @ text_features_max_[0][text_pos_idx,:]
                text_features_max_neg = text_neg_weights @ text_features_max_[1][text_neg_idx,:]
                prob_weights_.append(torch.stack([text_pos_weights, text_neg_weights], dim = 0))
                text_features_topk_.append(torch.stack([text_features_max_[0][text_pos_idx,:], text_features_max_[1][text_neg_idx,:]], dim = 0))

                samples_text_features.append(torch.stack([text_features_max_pos, text_features_max_neg], dim = 0))

            text_features_max = torch.stack(samples_text_features, dim = 0)
            prob_weights = torch.stack(prob_weights_, dim = 0)
            text_features_topk = torch.stack(text_features_topk_, dim = 0).view(-1, text_features_topk_[0].shape[-1])

            text_features_max_norm = text_features_max/text_features_max.norm(dim=-1, keepdim=True)
          
            text_probs = torch.bmm(text_features_max_norm, image_features_norm.unsqueeze(-1)).squeeze(-1)

            # EOT
            text_probs_ot, similarity_ot = OptimalTransport(args, (patch_features[-1]/patch_features[-1].norm(dim = -1, keepdim = True))[:,1:,:], 
                                       torch.stack(torch.chunk(text_features_topk/text_features_topk.norm(dim=-1, keepdim=True), dim = 0, chunks = 2), dim = 1), 
                                       entropy = img_entropy_map,
                                       prob = None)
            if text_probs_ot is None:
                continue
            probs = (text_probs + text_probs_ot)/0.07
            image_loss = F.cross_entropy(probs, label.long().cuda())
            image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity_ori, _ = get_similarity_map(patch_feature, text_features_max_norm)
                    similarity = ((similarity_ori+similarity_ot)/0.07).softmax(-1)
                    if vis_prompts is not None:
                        similarity_map = get_similarity_map(similarity[:, :-args.n_vis_ctx, :], args.image_size)
                    else:
                        similarity_map = get_similarity_map(similarity[:, :, :], args.image_size)
                    similarity_map_list.append(similarity_map.permute(0, 3, 1, 2))

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            (loss+image_loss+ortho_loss).backward(retain_graph=True)
            optimizer.step()
            loss_list.append(loss.item())
        
        # logs
        if (epoch + 1) % args.print_freq == 0:
            pos_fre_np = (pos_fre / pos_fre.sum()).detach().cpu().numpy()
            neg_fre_np = (neg_fre / neg_fre.sum()).detach().cpu().numpy()
            pos_fre_formatted = ', '.join([f'{item:.3f}' for item in pos_fre_np])
            neg_fre_formatted = ', '.join([f'{item:.3f}' for item in neg_fre_np])
            logger.info(f'epoch [{epoch + 1}/{args.epoch}], '
                        f'loss: {np.mean(loss_list):.4f}, '
                        f'image_loss: {np.mean(image_loss_list):.4f}, '
                        f'orth_loss: {np.mean(ortho_loss_list):.4f} \n'
                        f'fre: [{pos_fre_formatted}]/[{neg_fre_formatted}]')
        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_pool": prompt_pool.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("FGPT", add_help=True)

    # dataset setting
    parser.add_argument("--train_data_path", type=str, default="./path/to/dataset", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    
    # hyperpraameters of FGPT
    parser.add_argument("--n_vis_ctx", type=int, default=0, help="number of learnable visual prompts")
    parser.add_argument("--num_n", type=int, default=5, help="number of learnable normal text prompts")
    parser.add_argument("--num_a", type=int, default=5, help="number of learnable abnormal text prompts")
    parser.add_argument("--top_k", type=int, default=2, help="top k selection")
    parser.add_argument("--eps", type=float, default=0.1, help="epsion for Sinkhorn")
    parser.add_argument("--ot_threshold", type=float, default=1e-2, help="threshold for Sinkhorn")

    parser.add_argument("--epoch", type=int, default=30, help="epochs")

    # baseline setting
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")

    parser.add_argument("--n_ctx", type=int, default=12, help="length of learnable text prompt")
    parser.add_argument("--depth", type=int, default=9, help="depth of learnable text prompt")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="length of learnable token")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
