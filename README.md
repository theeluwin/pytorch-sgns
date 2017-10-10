# PyTorch SGNS

SkipGramNegativeSampling

Yet another but quite general [negative sampling loss](https://arxiv.org/abs/1310.4546) implemented in [PyTorch](http://www.pytorch.org). Corpus reference: [dl4j](https://deeplearning4j.org/word2vec).

It can be used with any embedding scheme! Pretty fast, I bet.

```python
V = len(vocab)
word2vec = Word2Vec(V=V)
sgns = SGNS(V=V, embedding=word2vec, batch_size=128, window_size=4, n_negatives=5)
for batch, (iword, owords) in enumerate(dataloader):
    loss = sgns(iword, owords)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
