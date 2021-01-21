from transformer_asr.models import get_model
from transformer_asr.dataset import AsrDataset, get_max_len
import tensorflow as tf

# model = get_model("hfl/chinese-roberta-wwm-ext")
#
# model.summary()


# max_in, max_out

# max_in, max_out = get_max_len("./cache/max_len.json", "hfl/chinese-roberta-wwm-ext",["./dev.tsv", "test.tsv", "train.tsv"])
max_in, max_out = get_max_len("./cache/max_len.json",
                              "hfl/chinese-roberta-wwm-ext")  # ["./dev.tsv", "test.tsv", "train.tsv"])

speech_config = {
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "feature_type": "log_mel_spectrogram",
    "preemphasis": 0.97,
    "normalize_signal": True,
    "normalize_feature": True,
    "normalize_per_feature": False
}

data = AsrDataset(batch_size=8,
                  tfrecords_dir="./cache",
                  max_input_len=max_in,
                  max_output_len=max_out,
                  source_file="./dev.tsv",
                  stage="dev",
                  speech_config=speech_config,
                  tokenizer_name="hfl/chinese-roberta-wwm-ext")

d = data.build_dataset()


def build_model():
    inputs_embeds = tf.keras.layers.Input(shape=(None, 80), dtype=tf.float32, name="audio")
    # x_mask = tf.keras.layers.Input(shape=(None, 80), dtype=tf.float32, name="audio_mask")

    # x = tf.keras.layers.Attention()([inputs_embeds, x_mask])

    x = tf.keras.layers.Attention()([inputs_embeds, inputs_embeds])
    x = tf.keras.layers.Add()([inputs_embeds, x])
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Conv1D(80, 42, strides=41)(x)

    x = tf.keras.layers.Dense(21128, activation="softmax", name="predict")(x)

    model = tf.keras.Model(inputs_embeds, x)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-2,
        decay_steps=1000,
        decay_rate=0.9
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),

        loss=tf.keras.losses.mean_squared_error,
        metrics=['accuracy']
    )
    model.summary()
    return model


model = build_model()
model.fit(d, epochs=2)
