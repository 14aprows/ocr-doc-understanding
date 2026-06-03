class TextEncoder:
    def __init__(self):
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:/-() "
        self.blank_idx = 0
        self.char2idx = {
            char: idx + 1 
            for idx, char in enumerate(self.chars)
        }
        self.idx2char = {
            idx + 1: char
            for idx, char in enumerate(self.chars)
        }
        self.idx2char[self.blank_idx] = ""

    def num_classes(self):
        return len(self.chars) + 1

    def encode(self, text):
        encoded = []
        for char in text:
            if char in self.char2idx:
                encoded.append(self.char2idx[char])
        return encoded

    def decode(self, indices):
        text = ""
        previous_idx = None

        for idx in indices:
           if idx != self.blank_idx and idx != previous_idx:
               text += self.idx2char.get(idx, "")
           previous_idx = idx
        return text

    def decode_batch(self, predictions):
        texts = []
        for pred in predictions:
            if hasattr(pred, "tolist"):
                pred = pred.tolist()
            texts.append(self.decode(pred))
        return texts