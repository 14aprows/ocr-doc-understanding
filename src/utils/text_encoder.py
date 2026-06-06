class TextEncoder:
    def __init__(self):
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:/-()'\"#&+%?!;_ "
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

    def normalize_text(self, text):
        text = str(text)
        text = text.lower()
        return text

    def encode(self, text):
        text = self.normalize_text(text)
        encoded = []
        for char in text:
            if char in self.char2idx:
                encoded.append(self.char2idx[char])
        return encoded

    def decode(self, indices):
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
            
        text = ""
        previous_idx = None

        for idx in indices:
           if idx != self.blank_idx and idx != previous_idx:
               text += self.idx2char.get(idx, "")
           previous_idx = idx
        return text

    def decode_logits(self, logits):
        predictions = logits.argmax(2)
        predictions = predictions.permute(1,0)
        return self.decode_batch(predictions)

    def decode_batch(self, predictions):
        texts = []
        for pred in predictions:
            texts.append(self.decode(pred))
        return texts

    def unknown_char(self, text):
        text = self.normalize_text(text)
        unknown = []
        for char in text:
            if char not in self.char2idx:
                unknown.append(char)
        return sorted(set(unknown))