"""Load the locked ContentVec encoder into Transformers HuBERT without legacy fairseq.

The 12-layer encoder uses the original ContentVec weights. Only pretraining heads
(label embeddings and final projection) are omitted; every inference tensor is checked.
Mapping follows transformers/models/hubert/convert_hubert_original_pytorch_checkpoint_to_pytorch.py.
"""
import pickle
import re
import types
from pathlib import Path
from ..common.paths import ROOT


class _UnusedDictionary:
    """Checkpoint vocabulary metadata; never used for acoustic inference."""
    pass


class _ContentVecUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('fairseq.data.dictionary', 'Dictionary'): return _UnusedDictionary
        return super().find_class(module, name)


def convert_state(original):
    out = {}; omitted = []
    for key, value in original.items():
        name = key
        if key in {'label_embs_concat', 'final_proj.weight', 'final_proj.bias'}:
            omitted.append(key); continue
        if key == 'mask_emb': name = 'masked_spec_embed'
        elif key.startswith('feature_extractor.conv_layers.'):
            name = re.sub(r'(feature_extractor.conv_layers\.\d+)\.0\.', r'\1.conv.', key)
            name = re.sub(r'(feature_extractor.conv_layers\.\d+)\.2\.', r'\1.layer_norm.', name)
        elif key.startswith('post_extract_proj.'): name = key.replace('post_extract_proj.', 'feature_projection.projection.')
        elif key.startswith('layer_norm.'): name = 'feature_projection.'+key
        elif key.startswith('encoder.pos_conv.0.'):
            name = key.replace('encoder.pos_conv.0.', 'encoder.pos_conv_embed.conv.')
            name = name.replace('weight_g', 'parametrizations.weight.original0').replace('weight_v', 'parametrizations.weight.original1')
        elif key.startswith('encoder.layers.'):
            name = key.replace('.self_attn.', '.attention.').replace('.self_attn_layer_norm.', '.layer_norm.')
            name = name.replace('.fc1.', '.feed_forward.intermediate_dense.').replace('.fc2.', '.feed_forward.output_dense.')
        if name in out: raise ValueError('Duplicate ContentVec tensor: '+name)
        out[name] = value.float()
    return out, omitted


class ContentVec:
    def __init__(self, device='cuda', variant='vec768l12'):
        import torch
        from transformers import HubertConfig, HubertModel
        if variant not in {'vec768l12', 'vec256l9'}:
            raise ValueError('Unsupported ContentVec variant: '+variant)
        self.variant = variant
        mod = types.ModuleType('contentvec_pickle'); mod.Unpickler = _ContentVecUnpickler; mod.load = pickle.load
        payload = torch.load(ROOT/'models/SoVitsSVC/contentvec.pt', map_location='cpu', pickle_module=mod, weights_only=False)
        cfg = payload['cfg']['model']
        if (cfg['encoder_layers'], cfg['encoder_embed_dim'], cfg['extractor_mode'], cfg['layer_norm_first']) != (12,768,'default',False):
            raise ValueError('Unexpected ContentVec architecture')
        config = HubertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
            intermediate_size=3072, hidden_act='gelu', feat_extract_norm='group', conv_bias=False,
            conv_dim=[512]*7, conv_stride=[5,2,2,2,2,2,2], conv_kernel=[10,3,3,3,3,2,2],
            num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16, do_stable_layer_norm=False,
            apply_spec_augment=False, hidden_dropout=0.1, attention_dropout=0.1,
            activation_dropout=0.0, feat_proj_dropout=0.1, layer_norm_eps=1e-5)
        config._attn_implementation = 'eager'
        self.model = HubertModel(config)
        state, omitted = convert_state(payload['model'])
        current = self.model.state_dict()
        if set(current) != set(state):
            raise ValueError(f'ContentVec mapping incomplete: missing={set(current)-set(state)}, extra={set(state)-set(current)}')
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.projection = None
        if variant == 'vec256l9':
            # The matching 4.1 vec256l9 base uses the original layer-9 final projection.
            self.projection = torch.nn.Linear(768, 256)
            self.projection.load_state_dict({
                'weight': payload['model']['final_proj.weight'].float(),
                'bias': payload['model']['final_proj.bias'].float()}, strict=True)
            self.projection.eval().requires_grad_(False).to(device)
            omitted = [key for key in omitted if not key.startswith('final_proj.')]
        self.device = device
        self.loading = {'encoder_tensors':len(state), 'omitted_training_heads':omitted,
                        'implementation':'transformers.HubertModel',
                        'projection_tensors': 2 if self.projection is not None else 0,
                        'output_layer': 9 if self.projection is not None else 12}

    def encoder(self, waveform):
        import torch
        with torch.inference_mode():
            # Original checkpoint task.normalize is False; preserve the raw 16 kHz waveform.
            wave = waveform.reshape(1,-1).to(self.device)
            if self.projection is not None:
                hidden = self.model(wave, output_hidden_states=True).hidden_states[9]
                return self.projection(hidden).transpose(1,2)
            return self.model(wave).last_hidden_state.transpose(1,2)
