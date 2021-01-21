from transformer_asr.speech_featurizers import TFSpeechFeaturizer, read_raw_audio

from transformers import BasicTokenizer, AutoTokenizer
from tqdm import tqdm
import numpy as np
import sys
import tensorflow as tf
import json
import multiprocessing


def float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def int64_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def print_one_line(*args):
    tf.print("\033[K", end="")
    tf.print("\r", *args, sep="", end=" ", output_stream=sys.stdout)


def bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


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


def get_max_len(cache_path, tokenizer_name, source_file_list=None):
    """

    :param source_file_list: tsv file list [audio_path\tduration\ttranscript]
    :param cache_path: json file max_len cache_path suggest to save together with tf_record
    :param tokenizer_name: huggingface tokenizer_name
    :return:
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tf.io.gfile.exists(cache_path):
        with tf.io.gfile.GFile(cache_path, mode='r') as gf:
            obj = json.load(gf)
            max_input_len = int(obj["max_input_len"])
            max_output_len = int(obj["max_output_len"])

    else:
        assert source_file_list, cache_path + " not exist source_file_list can not be null"

        max_input_len = 0
        max_output_len = 0

    if source_file_list is None:
        return max_input_len, max_output_len

    for source_file in source_file_list:

        lines = read_entries(source_file)

        for line in tqdm(lines, desc="[compute max len ]"):
            duration, text = line[1], line[2]
            audio_feature_len = int(float(duration) * 100 // 1) + 1
            txt_len = len(tokenizer.encode(text))

            if audio_feature_len > max_input_len:
                max_input_len = audio_feature_len
            if txt_len > max_output_len:
                max_output_len = txt_len

        print("max_input_len", max_input_len)
        print("max_output_len", max_output_len)
    with tf.io.gfile.GFile(cache_path, mode='w') as gf:
        json.dump(
            {
                "max_input_len": max_input_len,
                "max_output_len": max_output_len
            },
            gf)

    return max_input_len, max_output_len


def read_entries(source_file, stage="dev"):
    """

    :param source_file:  tsv file [audio_path\tduration\ttranscript]
    :param stage:
    :return:
    """
    with tf.io.gfile.GFile(source_file, "r") as f:
        lines = f.read().splitlines()
    # The files is "\t" seperated
    lines = lines[1:]
    lines = [line.split("\t", 2) for line in lines]

    lines = np.array(lines)
    if stage == "train":
        np.random.shuffle(lines)  # Mix transcripts.tsv

    return lines


class AsrDataset:

    def __init__(self,
                 batch_size: int,
                 tfrecords_dir: str,
                 max_input_len: int,
                 max_output_len: int,
                 source_file: str = None,
                 stage: str = "train",
                 speech_config: dict = None,
                 speech_featurizer: TFSpeechFeaturizer = None,
                 tokenizer_name: str = None,
                 tokenizer: BasicTokenizer = None,
                 tfrecords_shards=16,
                 buffer_size=2048
                 ):
        """

            :param buffer_size:
            :param tfrecords_shards:
            :param batch_size: batch_size for one device
            :param tfrecords_dir: gcs fold or local fold
            :param source_file: tsv file include audio-path,duration,text
            :param stage: train dev test
            :param max_input_len: if none ,will compute
            :param max_output_len: if none ,will compute
            :param speech_config: if none ,will use speech_featurizer
                eg:
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

            :param speech_featurizer: if none ,will use speech_config
            :param tokenizer_name: if none ,will use tokenizer
            :param tokenizer:  if none ,will use tokenizer_name
            :return:
            """

        assert not speech_config or not speech_featurizer
        assert not tokenizer_name or not tokenizer
        assert speech_config or speech_featurizer
        assert tokenizer_name or tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.source_file = source_file
        self.stage = stage
        self.tfrecords_dir = tfrecords_dir
        self.tfrecords_shards = tfrecords_shards
        if not self.tfrecords_dir.endswith("/"):
            self.tfrecords_dir = self.tfrecords_dir + "/"
        if speech_config:
            self.speech_featurizer = TFSpeechFeaturizer(speech_config)

        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.token_num = self.tokenizer.vocab_size

    def create_tfrecords(self):

        assert self.source_file, "source_file can not be null"
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

        print(f"Creating {self.stage}.tfrecord ...")

        entries = read_entries(self.source_file, self.stage)
        assert len(entries) > 0

        def get_shard_path(shard_id):
            return self.tfrecords_dir + f"{self.stage}_{shard_id}.tfrecord"

        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_entries = np.array_split(entries, self.tfrecords_shards)

        with multiprocessing.Pool(self.tfrecords_shards) as pool:
            pool.map(write_tfrecord_file, zip(shards, splitted_entries))

    def preprocess(self, audio_byte, transcript):
        with tf.device("/CPU:0"):
            audio = read_raw_audio(audio_byte, self.speech_featurizer.sample_rate)

            audio = self.speech_featurizer.extract(audio)
            # print(audio.shape)

            audio = tf.convert_to_tensor(audio, tf.float32)
            audio = tf.squeeze(audio)
            # label = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(transcript.decode()))
            label = self.tokenizer.encode(transcript.decode())

            label = tf.one_hot(label, self.token_num)

            label = tf.convert_to_tensor(label, tf.float32)

            audio_mask = tf.ones(audio.shape)

            return audio, audio_mask, label

    @tf.function
    def parse(self, record):
        feature_description = {
            "audio": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "transcript": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)

        return tf.numpy_function(
            self.preprocess,
            inp=[example["audio"], example["transcript"]],
            Tout=[tf.float32, tf.float32, tf.float32]
        )

    @tf.function
    def reshape(self, audio, audio_mask, label):

        # return {"audio": audio, "audio_mask": audio_mask, "predict": label}
        return [audio, label]

    def build_dataset(self):

        pattern = self.tfrecords_dir + f"{self.stage}*.tfrecord"

        file_list = tf.io.gfile.glob(pattern)

        if len(file_list) != self.tfrecords_shards:
            self.create_tfrecords()

        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB',
                                          num_parallel_reads=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        input_shape_for_tpu = self.speech_featurizer.shape
        input_shape_for_tpu[0] = self.max_input_len  # all samples should be padded to this length statically for TPU
        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=(
                tf.TensorShape(input_shape_for_tpu),
                tf.TensorShape(input_shape_for_tpu),
                tf.TensorShape([self.max_output_len, self.token_num])
            ),
            padding_values=(0., 0., 0.),
            drop_remainder=True
        )

        dataset = dataset.map(self.reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()
        return dataset
