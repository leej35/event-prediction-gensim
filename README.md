# Discrete Event Prediction in Time-Series with Gensim
Jeongmin Lee (jlee@cs.pitt.edu) / Nov 28. 2017

## Overview
This is a brief note on how one can utilize Gensim's CBOW implementation for Discrete Event Prediction.

To utilize Gensim's CBOW for your own task, you need to tweak (perhaps) two parts: data feeding part and model training part.

## Tweak Data Feeding: Having  My (own) Sentence Class
Data feeding in Gensim is done with its `sentence` class. Basically, training instance consists of sentences which are sequence of words. Gensim has its own sentence class that encapsulates sentences for training. 

We can create our own sentence class that feeds our own training instances like this:
```
class MySentences(object):
    def __init__(self, data_list):
        self.data = data_list
    def __repr__(self):
        return repr(self.data)
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        for instance in self.data:
            yield instance
```

With MySentence class, you can create the my_sentences object. We assume each alphabet represents an event type and single instance is a tuple consists of a list of context events and a single target event. For example, an instance of (['a','b'], 'z') represents that two events ['a','b'] are context event (preceding event) of an event 'z'.

You might want to have your own elaborated data feed MySentences class that parses your time series data into the list of list of these tuples. 

```
data_as_list = [ (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['a','b'], 'z'), (['e','f','g'], 'c'), (['e','f','h','i'], 'c')   ]
my_sentences = MySentences(data_as_list)
```


## Tweak Model Training Part
Basically, Gensim only supports processing its input as a list of sentences that each consists of a sequence of words. To facilitate our time-series instance that a tuple of context events and a target event, we need to change some part of Gensim's model training code. From now and on, we assume a sentence as a single train instance that consists of context events and a target event.

Gensim's word2vec provides two models: Skip-gram and CBOW. For us, we only consider CBOW. 

On high-level view, Gensim's CBOW has implemented in two parts: building vocabulary and training. Building vocabulary is done within build_vocab() function. Within it, we need to change scan_vocab() function that decouples context events and target events in a sentence. 

On training side, train() will manage jobs and train_batch_cbow() is the function that opens a data instance (sentence) and call subroutine function train_cbow_pair() that gets target and context events pair. We need to change train_batch_cbow() to facilitate process pairs of contexts and a target event instead of the sequence of words.


```
class Word2Vec __init__ (line 436)
  -> build_vocab()
    -> scan_vocab() <**change**> (line 683)
    -> scale_vocab()
    -> finalize_vocab()

  -> train(sentences) (line 908)
    -> job_producer() <**change**> (line 995)
    -> worker_loop() (line 980)
      -> _do_train_job() (line 901)
        -> train_batch_cbow() <**change**> (line 176)
          -> train_cbow_pair()
```


### Change code
Followings are how we actually changed the code to facilitate contexts and a target event data.

All Word2Vec (which includes CBOW) of Gensim is in `gensim/models/word2vec.py` file. 

On Github: <https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py>

### train_batch_cbow (line 178)
```
    def train_batch_cbow(model, sentences, alpha, work=None, neu1=None, compute_loss=False):
        """
        Update CBOW model by training on a sequence of sentences,
        where each sentence as tuple of (context_words and a target_word)
        """
        result = 0
        for sentence in sentences:
            context_words, target_word = sentence

            target_vocab = model.wv.vocab[target_word] 
            context_word_vocabs = [model.wv.vocab[w] for w in context_words if w in model.wv.vocab]

            context_indices = [word2.index for word2 in context_word_vocabs]
            l1 = np_sum(model.wv.syn0[context_indices], axis=0)  # 1 x vector_size
            if context_indices and model.cbow_mean:
                l1 /= len(context_indices)

            train_cbow_pair(model, target_vocab, context_indices, l1, alpha, compute_loss=compute_loss)

            result += len(context_word_vocabs) + 1

        return result

```

### scan_vocab (line 683)
```
    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):

            # sentence:  of context words and a target word
            context_words, target_word = sentence

            for word in context_words:
                vocab[word] += 1

            vocab[target_word] += 1    
            
            total_words += len(context_words) + 1

            if sentence_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                    sentence_no, total_words, len(vocab)
                )

        logger.info(
            "collected %i word types from a corpus of %i raw words and %i sentences",
            len(vocab), total_words, sentence_no + 1
        )
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab
        return total_words

```

### job_producer (line 995)
```
        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = start_alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warning("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):

                context_words, target_word = sentence

                sentence_length = self._raw_word_count([context_words]) + 1

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha
                    )
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if end_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
                        next_alpha = max(end_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

```




### (To run it) Modify the word2vec.py and install Gensim with modified version
1. Clone the Gensim from Github `git clone https://github.com/RaRe-Technologies/gensim.git`
2. Replace original `gensim/models/word2vec.py` file with `word2vec.py` in this repository. 
3. (At terminal) Go to the root of the Gensim repository and install `python setup.py install --user`


### Train the model

Let's define a model and train it with my_sentences object we created. 
```
model = gensim.models.Word2Vec(my_sentences, 
            sg=0,
            min_count=0, 
            size=3,  
            workers=4)

model.train(my_sentences, 
            total_examples=len(my_sentences),
            epochs=1000)
```
You can change hyperparameters as you like:
* `sg`=0 is CBOW. `sg`=1 is Skipgram.  **sg must be 0**
* `size` is the dimensionality of the feature vectors. 
* `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).
* `min_count` = ignore all words with total frequency lower than this.


Check vocabularies:
```
print(model.wv.vocab)
```

Get nearest neighbor:
```
query = ['a','b']
model.most_similar(query, topn=2)
```

