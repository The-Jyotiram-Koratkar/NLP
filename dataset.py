from pathlib import Path

class Dataset:
    def __init__(self):
        self.train_dir = Path('data/raw/aclImdb/train')
        self.test_dir = Path('data/raw/aclImdb/test')

    def _get_set(self, limit, directory):
        x = []
        y = []
        for file in (directory / 'pos').iterdir():
            if limit is not None and len(x) >= limit / 2:
                break
            x.append(file.read_text())
            y.append(1)
        for file in (directory / 'neg').iterdir():
            if limit is not None and len(x) >= limit:
                break
            x.append(file.read_text())
            y.append(0)
        return x, y

    def get_train_set(self, limit):
        return self._get_set(limit=limit, directory=self.train_dir)

    def get_test_set(self, limit):
        return self._get_set(limit=limit, directory=self.test_dir)
