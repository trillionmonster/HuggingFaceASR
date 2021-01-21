from transformer_asr.models import get_model


model = get_model("hfl/chinese-roberta-wwm-ext")

model.summary()

# model.variables

