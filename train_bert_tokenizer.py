from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False, # Must be False if cased model
            lowercase=True,
            wordpieces_prefix="##"
            )
path = '../data'
tokenizer.train(files=['../data/wiki/train.txt', '../data/news/train.txt', 
    '../data/twitter/train.txt', '../data/books/train.txt'], vocab_size=30000)
tokenizer.save_model(".", "bert")
