from .adapter_roberta import (RobertaForSequenceClassificationWithAdapter, 
                             RobertaModelWithAdapter,
                             RobertaWithAdapterConfig)


class XLMRobertaWithAdapterConfig(RobertaWithAdapterConfig):
    model_type = "xlm-roberta"


class XLMRobertaModelWithAdapter(RobertaModelWithAdapter):
    config_class = XLMRobertaWithAdapterConfig


class XLMRobertaForSequenceClassificationWithAdapter(RobertaForSequenceClassificationWithAdapter):
    config_class = XLMRobertaWithAdapterConfig