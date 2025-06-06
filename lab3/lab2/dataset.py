from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers


tokenizer = Tokenizer(BPE())

tokenizer.normalizer = Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=8000, show_progress=True)

files = ["/root/autodl-tmp/lab/dataset/shakesspeare/train/tiny-shakespeare-train.txt"]
tokenizer.train(files, trainer)

tokenizer.save("bpe-tokenizer.json")


if __name__ == '__main__':
    # 加载训练好的 tokenizer
    tokenizer = Tokenizer.from_file("bpe-tokenizer.json")

    # 测试编码
    output = tokenizer.encode("hello world")
    print("Tokens:", output.tokens)
    print("IDs:", output.ids)
    print("VOCAB SIZE:", tokenizer.get_vocab_size())