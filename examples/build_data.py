from transformer_asr.dataset import AsrDataset, get_max_len

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

data = AsrDataset(batch_size=64,
                  tfrecords_dir="./cache",
                  max_input_len=max_in,
                  max_output_len=max_out,
                  source_file="./dev.tsv",
                  stage="dev",
                  speech_config=speech_config,
                  tokenizer_name="hfl/chinese-roberta-wwm-ext")

d = data.build_dataset()

print(next(iter(d)))
