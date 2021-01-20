import tensorflow as tf
from transformers import TFAutoModelForMaskedLM

from transformers import TFBertMainLayer


###"hfl/chinese-roberta-wwm-ext"

def get_model(pretrain_name,
              initial_learning_rate=1e-5
              ):
    model = TFAutoModelForMaskedLM.from_pretrained(pretrain_name, from_pt=True)

    encoder = TFBertMainLayer(config=model.config, name="encoder")

    decoder = model.mlm
    decoder.trainable = False

    inputs_embeds = tf.keras.layers.Input(shape=(None, 80), dtype=tf.float32, name="audio")
    x_mask = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="audio_mask")

    x = tf.keras.layers.Dense(model.config.hidden_size, name="audio_convert")(inputs_embeds)

    out = encoder(None, attention_mask=x_mask, inputs_embeds=x)
    out = decoder(out.last_hidden_state)

    model_asr = tf.keras.Model([inputs_embeds, x_mask], out)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )
    model_asr.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=['sparse_categorical_accuracy']
    )

