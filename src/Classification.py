from pathlib import Path
import csv

import numpy as np
import tflite_runtime.interpreter as tflite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "csv" / "yamnet.tflite"
DEFAULT_CLASS_MAP_PATH = PROJECT_ROOT / "csv" / "yamnet_class_map.csv"


def load_class_names(path=DEFAULT_CLASS_MAP_PATH):
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        return [class_name for _, _, class_name in reader]


class SoundClassifier:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, class_map_path=DEFAULT_CLASS_MAP_PATH):
        self.model = tflite.Interpreter(model_path=str(model_path))
        self.class_names = load_class_names(class_map_path)

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        self.waveform_input_index = input_details[0]["index"]
        self.scores_output_index = output_details[0]["index"]
        self.embeddings_output_index = output_details[1]["index"]
        self.spectrogram_output_index = output_details[2]["index"]

    def tonumpy(self, buffers):
        chunks = [
            np.frombuffer(buffer, dtype=np.int32).astype(np.float32)
            for buffer in buffers
            if buffer
        ]
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def preprocess(self, data):
        if data.size == 0:
            return data

        peak = np.max(np.abs(data))
        if peak > 1:
            return data / peak
        return data

    def classifier(self, data):
        if data.size < 2:
            return ""

        waveform = np.delete(data, 1).astype(np.float32)
        self.model.resize_tensor_input(self.waveform_input_index, [len(waveform)])
        self.model.allocate_tensors()
        self.model.set_tensor(self.waveform_input_index, waveform)
        self.model.invoke()

        scores = self.model.get_tensor(self.scores_output_index)
        mean_scores = scores.mean(axis=0)
        return self.class_names[int(mean_scores.argmax())]


Classificate = SoundClassifier
