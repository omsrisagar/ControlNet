import json
import os.path

import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.data_dir = data_dir
        self.data = self._list_image_files_recursively(self.data_dir)
        with open(os.path.join(self.data_dir, 'prompt.json'), 'r') as f:
            self.prompt_dict = json.load(f)
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        # with open(os.path.join(self.data_dir, 'prompt.json'), 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))

    @classmethod
    def _list_image_files_recursively(self, data_dir, return_full_paths=False):
        results = []
        for entry in sorted(os.listdir(data_dir)):
            entry = os.path.join(data_dir, entry) if return_full_paths else entry
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(entry)
            # elif bf.isdir(full_path):
            #     results.extend(_list_image_files_recursively(full_path))
        return results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        target_filename = os.path.join(self.data_dir, path)
        source_filename = os.path.join(self.data_dir.replace('_img', '_sketch', 1), path)
        # source_filename = item['source']
        # target_filename = item['target']
        # prompt = item['prompt']
        # prompt = "A high-quality, detailed, and professional image"
        prompt = self.prompt_dict[path]

        # source = cv2.imread(os.path.join(self.data_dir, source_filename))
        # target = cv2.imread(os.path.join(self.data_dir, target_filename))
        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

