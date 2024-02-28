# End to end Tokenization

This is the official repository for the EMNLP 2023 paper: [Learn Your Tokens: Word-Pooled Tokenization for Language Modeling](https://aclanthology.org/2023.findings-emnlp.662/).

## Example usage
The model is defined in `newmodel.py`.

```bash
python unitrain.py -dataset shakespeare -base sub --no-e2e --device 2
```
It serves as an end-to-end tokenized implementation of [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy.

## Citation

Please consider citing us if you found our project useful:

```
Avijit Thawani, Saurabh Ghanekar, Xiaoyuan Zhu, and Jay Pujara. 2023.
Learn Your Tokens: Word-Pooled Tokenization for Language Modeling.
In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9883–9893, Singapore.
Association for Computational Linguistics.
```

```
@inproceedings{thawani-etal-2023-learn,
    title = "Learn Your Tokens: Word-Pooled Tokenization for Language Modeling",
    author = "Thawani, Avijit  and
      Ghanekar, Saurabh  and
      Zhu, Xiaoyuan  and
      Pujara, Jay",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.662",
    doi = "10.18653/v1/2023.findings-emnlp.662",
    pages = "9883--9893",
    abstract = "Language models typically tokenize text into subwords, using a deterministic, hand-engineered heuristic of combining characters into longer surface-level strings such as {`}ing{'} or whole words. Recent literature has repeatedly shown the limitations of such a tokenization strategy, particularly for documents not written in English and for representing numbers. On the other extreme, byte/character-level language models are much less restricted but suffer from increased sequence description lengths and a subsequent quadratic expansion in self-attention computation. Recent attempts to compress and limit these context lengths with fixed size convolutions is helpful but completely ignores the word boundary. This paper considers an alternative {`}learn your tokens{'} scheme which utilizes the word boundary to pool bytes/characters into word representations, which are fed to the primary language model, before again decoding individual characters/bytes per word in parallel. We find that our moderately expressive and moderately fast end-to-end tokenizer outperform by over {`}300{\%}{`} both subwords and byte/character models over the intrinsic language modeling metric of next-word prediction across datasets. It particularly outshines on rare words, outperforming by a factor of 30! We extensively study the language modeling setup for all three categories of tokenizers and theoretically analyze how our end-to-end models can also be a strong trade-off in efficiency and robustness.",
}
```


### License

MIT
