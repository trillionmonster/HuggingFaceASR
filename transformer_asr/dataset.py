from transformer_asr.speech_featurizers import TFSpeechFeaturizer, read_raw_audio
from transformer_asr.augments import Augmentation
from transformers import BasicTokenizer, AutoTokenizer
from tqdm import tqdm
import numpy as np
import sys
import tensorflow as tf
import json
import multiprocessing

augmentations = Augmentation()


def float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def int64_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def print_one_line(*args):
    tf.print("\033[K", end="")
    tf.print("\r", *args, sep="", end=" ", output_stream=sys.stdout)


def bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def build_dataset(
        batch_size,
        tfrecords_dir: str = None,
        source_file=None,
        stage="train",
        speech_config=None,
        speech_featurizer: TFSpeechFeaturizer = None,
        tokenizer_name=None,
        tokenizer: BasicTokenizer = None,
        tfrecords_shards=16,
        buffer_size=2048
):
    """

    :param device_num:
    :param buffer_size:
    :param tfrecords_shards:
    :param batch_size: batch_size for one device
    :param tfrecords_dir: gcs fold or local fold
    :param source_file: tsv file include audio-path,duration,text
    :param stage: train dev test
    :param max_input_len: if none ,will compute
    :param max_output_len: if none ,will compute
    :param speech_config: if none ,will use speech_featurizer

    speech_config = {
            "sample_rate": int,
            "frame_ms": int,
            "stride_ms": int,
            "num_feature_bins": int,
            "feature_type": str,
            "delta": bool,
            "delta_delta": bool,
            "pitch": bool,
            "normalize_signal": bool,
            "normalize_feature": bool,
            "normalize_per_feature": bool
        }
    :param speech_featurizer: if none ,will use speech_config
    :param tokenizer_name: if none ,will use tokenizer
    :param tokenizer:  if none ,will use tokenizer_name
    :return:
    """
    assert speech_config is None and speech_featurizer is None
    assert tokenizer_name is None and tokenizer is None
    assert speech_config and speech_featurizer
    assert tokenizer_name and tokenizer
    if not tfrecords_dir.endswith("/"):
        tfrecords_dir = tfrecords_dir + "/"
    if speech_config:
        speech_featurizer = TFSpeechFeaturizer(speech_config)

    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def read_entries():

        with tf.io.gfile.GFile(source_file, "r") as f:
            lines = f.read().splitlines()
        # The files is "\t" seperated
        lines = [line.split("\t", 2) for line in lines]
        lines = np.array(lines)
        if stage == "train":
            np.random.shuffle(lines)  # Mix transcripts.tsv

        return lines

    def compute_max_len():

        max_int_len = 0
        max_out_len = 0

        lines = read_entries()

        for line in tqdm(lines, desc="[compute max len ]"):
            duration, text = line[1], line[2]
            audio_feature_len = int(duration * 100 // 1) + 1
            txt_len = len(tokenizer.encode(text))

            if audio_feature_len > max_int_len:
                max_int_len = audio_feature_len
            if txt_len > max_out_len:
                max_out_len = txt_len

        print("max_int_len", max_int_len)
        print("max_out_len", max_out_len)

        return max_int_len, max_out_len

    def to_tfrecord(path, audio, transcript):
        feature = {
            "path": bytestring_feature([path]),
            "audio": bytestring_feature([audio]),
            "transcript": bytestring_feature([transcript])
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecord_file(splitted_entries):
        shard_path, entries = splitted_entries
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for audio_file, _, transcript in entries:
                with open(audio_file, "rb") as f:
                    audio = f.read()
                example = to_tfrecord(bytes(audio_file, "utf-8"), audio, bytes(transcript, "utf-8"))
                out.write(example.SerializeToString())
                print_one_line("Processed:", audio_file)
        print(f"\nCreated {shard_path}")

    def create_tfrecords():

        assert source_file, "source_file can not be null"
        if not tf.io.gfile.exists(tfrecords_dir):
            tf.io.gfile.makedirs(tfrecords_dir)

        print(f"Creating {stage}.tfrecord ...")

        entries = read_entries()
        assert len(entries) > 0

        def get_shard_path(shard_id):
            return tfrecords_dir + f"{stage}_{shard_id}.tfrecord"

        shards = [get_shard_path(idx) for idx in range(1, tfrecords_shards + 1)]

        splitted_entries = np.array_split(entries, tfrecords_shards)
        with multiprocessing.Pool(tfrecords_shards) as pool:
            pool.map(write_tfrecord_file, zip(shards, splitted_entries))

        max_input_len, max_output_len, = compute_max_len()

        with tf.io.gfile.GFile(tfrecords_dir + "max_input_len.json", mode='w') as gf:
            json.dump({
                "max_input_len": max_input_len,
                "max_output_len": max_output_len}, gf)

    def preprocess(audio_byte, transcript):
        with tf.device("/CPU:0"):
            audio = read_raw_audio(audio_byte, speech_featurizer.sample_rate)

            audio = augmentations.before.augment(audio)

            audio = speech_featurizer.extract(audio)

            audio = augmentations.after.augment(audio)

            audio = tf.convert_to_tensor(audio, tf.float32)

            label = tokenizer.encode(transcript)

            label = tf.convert_to_tensor(label, tf.int32)

            audio_mask = tf.ones(audio.shape[0])

            return audio, audio_mask, label

    @tf.function
    def parse(record):
        feature_description = {
            "audio": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "transcript": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)

        return tf.numpy_function(
            preprocess,
            inp=[example["audio"], example["transcript"]],
            Tout=[tf.float32, tf.int32, tf.int32]
        )

    pattern = tfrecords_dir + f"{stage}*.tfrecord"

    if not tf.io.gfile.exists(pattern):
        create_tfrecords()

    files_ds = tf.data.Dataset.list_files(pattern)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)
    dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB',
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE)

    if tf.io.gfile.exists(tfrecords_dir + "max_len.json"):
        with tf.io.gfile.GFile(tfrecords_dir + "max_len.json", mode='r') as gf:
            obj = json.load(gf)
            max_input_len = int(obj["max_input_len"])
            max_output_len = int(obj["max_output_len"])
    else:
        assert source_file, tfrecords_dir + "max_input_len.json not exist source_file can not be null"

        max_input_len, max_output_len = compute_max_len()

    dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

    input_shape_for_tpu = speech_featurizer.shape
    input_shape_for_tpu[0] = max_input_len  # all samples should be padded to this length statically for TPU training
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            tf.TensorShape(input_shape_for_tpu),
            tf.TensorShape([max_input_len]),
            tf.TensorShape([max_output_len])
        ),
        padding_values=(0., 0, tokenizer.pad_token_id),
        drop_remainder=True
    )

    return dataset
