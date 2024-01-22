from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaModel, \
    RobertaLayer, RobertaForSequenceClassification, RobertaClassificationHead
from transformers.modeling_utils import apply_chunking_to_forward
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig
import pickle

from .utils import label_smoothed_nll_loss, total_param_dim
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.0000001)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class RobertaWithAdapterConfig(BertConfig):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            adapter_dim=64,
            adapt_layer_norm=False,
            unfreeze_hyper_encoder=False,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
    

        # Adapter
        self.adapter_dim = adapter_dim
        self.generator_hdim = 64
        self.generator_hdim_small = 1
        self.adapt_layer_norm = adapt_layer_norm
        self.unfreeze_hyper_encoder = unfreeze_hyper_encoder


class ModelWithAdapter(nn.Module):
    
    def init_adapter(self, input_dim, mid_dim, output_dim, config):
        self.config = config
        self.adapter_name_to_weight = OrderedDict()
        self.skip_adapter = self.config.skip_adapter
        self.no_param_gen = self.config.no_param_gen
        self.adapter_down_weight, self.adapter_up_weight, self.adapter_down_bias, self.adapter_up_bias = \
            None, None, None, None
        #self.adapter_up, self.adapter_down = None, None  # modules
        self.adapter_id = 0
        if self.no_param_gen:
            self.all_adapters = nn.ModuleList()

        if self.no_param_gen:
            task_num = self.config.task_num
            for i in range(task_num):
                adapter = nn.ModuleList(
                    [nn.Linear(input_dim, mid_dim),
                    nn.Linear(mid_dim, output_dim)]
                ).cuda()
                self.all_adapters.append(adapter)
        #else:
        self.adapter_down_weight = torch.zeros(input_dim, mid_dim).cuda()
        self.adapter_down_bias = torch.zeros(mid_dim).cuda()
        self.adapter_up_weight = torch.zeros(mid_dim, output_dim).cuda()
        self.adapter_up_bias = torch.zeros(output_dim).cuda()
        self.dirty = False

    def set_adapter_down_weight(self, tensor):
        self.adapter_down_weight = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][0].weight.copy_(tensor)
            # self.adapter_down_weight = None
            # self.adapter_down_func = nn.Linear(tensor.size(0), tensor.size(1)).cuda()

    def set_adapter_down_bias(self, tensor):
        self.adapter_down_bias = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][0].bias.copy_(tensor)
            #self.adapter_down_bias = None

    def set_adapter_up_weight(self, tensor):
        self.adapter_up_weight = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][1].weight.copy_(tensor)
            #self.adapter_up_weight = None
            #self.adapter_up_func = nn.Linear(tensor.size(0), tensor.size(1)).cuda()

    def set_adapter_up_bias(self, tensor):
        self.adapter_up_bias = tensor
        if self.no_param_gen:
            self.all_adapters[self.adapter_id][1].bias.copy_(tensor)

    def set_adapter_id(self, adapter_id):
        self.adapter_id = adapter_id

    def register_adapter_name_to_weight(self, names, weights):
        for name, weight in zip(names, weights):
            self.adapter_name_to_weight[name] = weight

    def get_my_module_weight_dims(self):
        return [
            self.adapter_down_weight.size(),
            self.adapter_down_bias.size(),
            self.adapter_up_weight.size(),
            self.adapter_up_bias.size()
        ]

    def get_my_weight_dim(self):
        s = total_param_dim(self.get_my_module_weight_dims())
        return s

    def adapter_down(self, x):
        if self.no_param_gen:
            return self.all_adapters[self.adapter_id][0](x)
        return F.linear(x, self.adapter_down_weight.t(), self.adapter_down_bias)

    def adapter_up(self, x):
        if self.no_param_gen:
            return self.all_adapters[self.adapter_id][1](x)
        return F.linear(x, self.adapter_up_weight.t(), self.adapter_up_bias)

    def set_adapter_weights(self, weight_vector):
        # default behavior
        return self.set_my_adapter_weights(weight_vector)
            
    def set_my_adapter_weights(self, weight_vector):
        sizes = self.get_my_module_weight_dims()
        prev_start = 0
        for size, (name, value) in zip(sizes, self.adapter_name_to_weight.items()):
            flat_size = np.product(size)
            weight_data = weight_vector[prev_start:prev_start + flat_size].cuda()
            # value.copy_(weight_data.view(*value.size()))
            if not self.no_param_gen:
                setattr(self, name, weight_data.view(*value.size()))
            else:
                #getattr(self, name).data.copy_(weight_data.view(*value.size()).detach())
                weight = weight_data.view(*value.size())
                if name == 'adapter_down_weight':
                    self.all_adapters[self.adapter_id][0].weight.data.copy_(weight_data.view(*value.size()).t())
                elif name == 'adapter_down_bias':
                    self.all_adapters[self.adapter_id][0].bias.data.copy_(weight_data.view(*value.size()))
                elif name == 'adapter_up_weight':
                    self.all_adapters[self.adapter_id][1].weight.data.copy_(weight_data.view(*value.size()).t())
                elif name == 'adapter_up_bias':
                    self.all_adapters[self.adapter_id][1].bias.data.copy_(weight_data.view(*value.size()))
                else:
                    raise ValueError(name)
            prev_start += flat_size


class RobertaLayerWithAdapter(RobertaLayer, ModelWithAdapter):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.adapter_dim = config.adapter_dim
        self.init_adapter(self.embed_dim, self.adapter_dim, self.embed_dim, config)
        
        self.register_adapter_name_to_weight(['adapter_down_weight', 'adapter_down_bias','adapter_up_weight',
                                              'adapter_up_bias'],[self.adapter_down_weight, self.adapter_down_bias,
                                             self.adapter_up_weight, self.adapter_up_bias])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        if not self.skip_adapter:
            residual_adapter = self_attention_outputs
            self_attention_outputs = self.adapter_down(self_attention_outputs)
            self_attention_outputs = self.activation_fn(self_attention_outputs)
            self_attention_outputs = self.adapter_up(self_attention_outputs)
            self_attention_outputs = residual_adapter + self_attention_outputs
            
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

class RoberaEncodeWithAdapter(RobertaEncoder):
    def __init__(self, config: RobertaConfig, embed_tokens):
        super(RoberaEncodeWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [RobertaLayerWithAdapter(config) for _ in range(config.encoder_layers)]
        )


class RobertaModelWithAdapter(RobertaModel):
    def __init__(self, config: RobertaConfig):
        super(RobertaModelWithAdapter, self).__init__(config)
        self.encoder = RoberaEncodeWithAdapter(config, self.shared)


class RobertaForSequenceClassificationWithAdapter(RobertaForSequenceClassification, ModelWithAdapter):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RobertaModelWithAdapter(config, add_pooling_layer=False)
        
        if config.mtl:
            if len(config.tasks) != config.mtl_task_num:
                raise Exception(f"Number of tasks is not equal to mtl task number") 
            self.classification_heads_dict = nn.ModuleDict()
            for task in config.tasks:
                self.classification_heads_dict[task] = RobertaClassificationHead(config)
                self.reinit_classification_head(self.classification_heads_dict[task])
        else:
            self.classification_head = RobertaClassificationHead(config)
            self.reinit_classification_head(classification_head=self.classification_head)
        self.config = config
        self.config.sep_token_id = 50265
        
        # self.adapter_down_weight, self.adapter_down_bias, self.adapter_up_weight, self.adapter_up_bias = \
        #     None, None, None, None
        self.task_name_to_vocab_space = None
    
    def reinit_classification_head(self, classification_head):
        self.model._init_weights(classification_head.dense)
        self.model._init_weights(classification_head.out_proj)
    
    def set_classification_head(self, task_name):
        self.classification_head = self.classification_heads_dict[task_name]
    
    def get_children_adapter_modules(self):
        return [_ for _ in self.model.encoder.layers] + [_ for _ in self.model.decoder.layers]
    
    def get_adapter_dims(self):
        adapter_modules = self.get_children_adapter_modules()
        required_weight_dims = [module.get_my_weight_dim() for module in adapter_modules] #  + [self.get_my_weight_dim()]
        return required_weight_dims
    
    def get_classification_head_dims(self):
        classification_head = [p.size() for p in self.classification_head.parameters()]
        required_weight_dims = total_param_dim(classification_head)
        return required_weight_dims

    def set_adapter_weights(self, all_adapter_weights):
        all_children_adapter_modules = self.get_children_adapter_modules()
        for module, weight in zip(all_children_adapter_modules, all_adapter_weights[:len(all_children_adapter_modules)]):
            module.set_adapter_weights(weight)
        # wb_weights = torch.cat(all_adapter_weights[len(all_children_adapter_modules):], 0)
        self.set_classification_head_weights(all_adapter_weights[-1])
    
    def set_classification_head_weights(self, weight_vector):
        prev_start = 0
        for layer in [self.classification_head.dense.weight, self.classification_head.dense.bias, 
                      self.classification_head.out_proj.weight, self.classification_head.out_proj.bias]:
            flat_size = np.product(layer.size())
            weight_data = weight_vector[prev_start:prev_start + flat_size].cuda()
            layer.data.copy_(weight_data.view(*layer.size())) # DID NOT GET Transpose
            prev_start += flat_size
    
    def set_adapter_id(self, adapter_id):
        adapter_modules = self.get_children_adapter_modules()
        for module in adapter_modules:
            module.set_adapter_id(adapter_id)
        self.adapter_id = adapter_id

    def load_adapter_weights_from_path(self, path):
        with open(path,'rb') as f:
            weights = pickle.load(f)
        self.set_adapter_weights(weights)

    def set_label_vocab_space(self, valid_token_ids):
        self.valid_token_ids = valid_token_ids
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


