"""
Modified BLIP-2 Q-Former for Band Gap Prediction with ALIGNN
- CGCNN → ALIGNN
- ITG loss 제거
- Masked prediction loss 추가
"""

import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.models.blip2_models.blip2 import disabled_train
from lavis.models.blip_models.blip_outputs import BlipOutput
from model.blip2 import Blip2Base
from model.dist_funs import pl_concat_all_gather


class Blip2QformerBandGap(Blip2Base):
    """
    Modified BLIP-2 Q-Former for Band Gap Prediction

    Changes from original:
    1. CGCNN → ALIGNN encoder
    2. Remove ITG (image-to-text generation) loss
    3. Add Masked Prediction loss for band gap
    4. Template: "{source}_bandgap is [MASK] eV. {robocrys}"
    """

    def __init__(
        self,
        gtm,  # Graph-Text Matching
        bert_name,
        temperature,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        args=None,
    ):
        super().__init__()
        self.gtm = gtm
        self.args = args
        self.tokenizer = self.init_tokenizer()

        # Initialize ALIGNN encoder (instead of GIN/Uni-Mol)
        self.alignn_encoder, self.ln_graph = self.init_alignn_encoder(args)

        # Freeze or tune ALIGNN
        if not args.tune_gnn:
            for name, param in self.alignn_encoder.named_parameters():
                param.requires_grad = False
            self.alignn_encoder = self.alignn_encoder.eval()
            self.alignn_encoder.train = disabled_train
            logging.info("Freeze ALIGNN encoder")

        # Initialize Q-Former
        self.Qformer, self.query_tokens = self.init_Qformer(
            bert_name,
            num_query_token,
            self.alignn_encoder.feature_dim,  # ALIGNN feature dim
            cross_attention_freq
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # Projection layers
        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # GTM head
        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # Masked prediction head for band gap
        self.mask_prediction_head = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)  # Predict band gap value
        )

        self.temperature = temperature

    def init_alignn_encoder(self, args):
        """
        Initialize ALIGNN encoder

        Returns:
            alignn_encoder: ALIGNN model
            ln_graph: Layer normalization
        """
        from MoE.alignn.model import ALIGNNRegression

        alignn_config = {
            'alignn_layers': args.alignn_layers,
            'gcn_layers': args.gcn_layers,
            'hidden_features': args.hidden_features,
            'embedding_features': args.embedding_features,
            'output_features': 1,
            'dropout': 0.0,  # No dropout in encoder
        }

        alignn_encoder = ALIGNNRegression(**alignn_config)

        # Layer normalization for graph features
        ln_graph = nn.LayerNorm(args.hidden_features)

        # Set feature_dim attribute
        alignn_encoder.feature_dim = args.hidden_features

        return alignn_encoder, ln_graph

    def contrast(self, features_graph, features_text, return_sim=False):
        """
        ITC (Image-Text Contrastive) Loss

        Args:
            features_graph: [B, num_qs, D]
            features_text: [B, D]
        """
        batch_size = features_graph.size(0)

        # Normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # Cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze()
        sim_g2t, _ = sim_q2t.max(-1)  # [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def forward(self, batch):
        """
        Forward pass

        Args:
            batch: (graphs, lg_graphs, texts, band_gaps, sources)
                - graphs: DGL atom graphs
                - lg_graphs: DGL line graphs
                - texts: Robocrys descriptions (with [MASK])
                - band_gaps: Ground truth band gap values
                - sources: Source labels

        Returns:
            BlipOutput with losses
        """
        graphs, lg_graphs, texts, band_gaps, sources = batch

        # === 1. ALIGNN Encoding ===
        with torch.set_grad_enabled(self.tune_gnn):
            graph_embeds = self.alignn_encoder.get_features(graphs, lg_graphs)  # [B, D]
            graph_embeds = self.ln_graph(graph_embeds)

        graph_embeds = graph_embeds.unsqueeze(1)  # [B, 1, D]

        # === 2. Text Tokenization ===
        text_tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.args.max_text_len,
            return_tensors='pt'
        ).to(self.device)

        # === 3. Q-Former: Graph → Query Tokens ===
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(self.device),
            use_cache=True,
            return_dict=True,
        )
        graph_feats = query_output.last_hidden_state  # [B, num_q, D]

        # === 4. ITC Loss (Graph-Text Contrastive) ===
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = text_output.last_hidden_state[:, 0, :]  # [CLS] token

        graph_feats_proj = self.graph_proj(graph_feats)  # [B, num_q, embed_dim]
        text_feat_proj = self.text_proj(text_feat)  # [B, embed_dim]

        # Gather features from all GPUs
        graph_feats_proj_all = pl_concat_all_gather(graph_feats_proj)
        text_feat_proj_all = pl_concat_all_gather(text_feat_proj)

        loss_itc = self.contrast(graph_feats_proj_all, text_feat_proj_all)

        # === 5. ITM Loss (Graph-Text Matching) ===
        loss_itm = torch.tensor(0.0, device=self.device)
        if self.gtm:
            # Create positive and negative pairs
            text_input_ids_world = pl_concat_all_gather(text_tokens.input_ids)
            text_attention_mask_world = pl_concat_all_gather(text_tokens.attention_mask)
            graph_embeds_world = pl_concat_all_gather(graph_embeds)

            with torch.no_grad():
                sim_g2t = (graph_feats_proj @ text_feat_proj_all.t()).max(dim=1)[0]
                sim_t2g = (text_feat_proj @ graph_feats_proj_all.max(dim=1)[0].t())

                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4

            # Select negative samples
            graph_embeds_neg = []
            for b in range(graph_embeds.size(0)):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(graph_embeds_world[neg_idx])
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(text_tokens.input_ids.size(0)):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])
            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            # Combine positive and negative
            text_ids_all = torch.cat([text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0)
            text_atts_all = torch.cat([text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg], dim=0)
            graph_embeds_all = torch.cat([graph_embeds, graph_embeds_neg, graph_embeds], dim=0)

            query_tokens_itm = self.query_tokens.expand(graph_embeds_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(self.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(self.device),
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, :query_tokens_itm.size(1), :]
            vl_output = self.gtm_head(vl_embeddings).mean(dim=1)

            itm_labels = torch.cat([torch.ones(graph_embeds.size(0), dtype=torch.long),
                                   torch.zeros(2 * graph_embeds.size(0), dtype=torch.long)], dim=0).to(self.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels)

        # === 6. Masked Prediction Loss ===
        # Find [MASK] token positions
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (text_tokens.input_ids == mask_token_id)

        # Get [MASK] token embeddings from Q-Former
        mask_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(self.device),
            return_dict=True,
        )

        # Extract [MASK] token representations
        mask_embeds_list = []
        for i, has_mask in enumerate(mask_positions.any(dim=1)):
            if has_mask:
                mask_pos = mask_positions[i].nonzero(as_tuple=True)[0][0]
                # Use query tokens pooled representation
                mask_embed = mask_output.last_hidden_state[i, :query_tokens.size(1), :].mean(dim=0)
                mask_embeds_list.append(mask_embed)

        if len(mask_embeds_list) > 0:
            mask_embeds = torch.stack(mask_embeds_list)
            predicted_bandgaps = self.mask_prediction_head(mask_embeds).squeeze()

            # Get corresponding ground truth
            valid_bandgaps = band_gaps[mask_positions.any(dim=1)]

            # MSE loss for band gap prediction
            loss_mask = F.mse_loss(predicted_bandgaps, valid_bandgaps)
        else:
            loss_mask = torch.tensor(0.0, device=self.device)

        # === 7. Total Loss ===
        # ITC + ITM + Masked Prediction (NO ITG!)
        loss = loss_itc + loss_itm + loss_mask

        return BlipOutput(
            loss=loss,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_mask,  # Use loss_lm field for masked prediction
        )
