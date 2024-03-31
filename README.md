# NLP

## Hugging Face

### What exactly is natural language processing supposed to do?

 - Classification, Machine Translation, Sentiment Analysis, Intelligent Customer Service, Summarization & Reading Comprehension, etc...

 - The learning of our language skills comes from the little things in life, a conversation, a reading is learning.
 - So we need to train NLP models just to get a final output?

### How do you come to develop learning skills in modeling?
 - Do we need specific tasks and labels? Does every conversation we have correspond to a standard answer? It doesn't.
 - It is more important to train reading skills, learning skills, comprehension skills, then just give the model reading material.
 - The so-called reading material is our human text data, novels, news, movies, etc. are all possible.
 - So, what we need to model now is language comprehension, not categorization kind of specialized skills.

### Journeyman in NLP
 - Early NLP was simpler, with no trained learning ability at all, just domain-specific tasks.
 - Nowadays, NLP can be simply divided into two major families: the BERT family and the GPT family.

### What NLP really spells?
 - What about spelling out the network structure, the loss function, or various training techniques?
 - From the current NLP more core model, the main spell is the amount of data and the number of parameters.
 - The models that have been scrubbed, as well as the cooler ones, have horrendous amounts of training data and parameters.
 - What can we do? Do we have to train models too? Do we have to use huge amounts of data too?

### How do we get started with NLP?
 - Hand it over to Transformer.

### I'd like to welcome our protagonist today.

 - Huggingface is the best of all worlds, including all the core models of NLP today.
 - To fine-tune our own task, we simply process our data and continue to train the model.

### It's not just a toolkit, it's a community, and a stage for NLP bigwigs.

 - More and more academic bigwigs are using it to open source models to publicize papers as well as research results.

### Stories about it
 - Rumor has it that 30 part-time development and algorithm engineers have leveraged a $2 billion market cap.

 - In fact, this can not be separated from the power of open source, the AI field is so in need of a stage and community.

 - Time makes the hero, in time for Transformer to explode in the AI field, the first to eat the crab!

 - BERT and GPT Sweep NLP, Huggingface Reaps the Benefits, Community-Driven Technology Advancement.

### Two birds with one stone, divide and conquer.

 - Ai is not only academically driven but also engineered on the ground.

 - Academics come to provide models for the community to demonstrate their status and competence in the field.

 - The project is to complete their own tasks through the pre-trained models provided by the community, and the efficiency of the project on the ground is very high.

## Transformer
### What's the one thing to do?
 - The basic composition is still the Seq2Seq network, which is common in machine translation modeling.

 - The inputs and outputs are intuitive, and the core architecture is the network design in the middle.


![](/assets/Image(1).png)

### Traditional RNN networks
 - What's wrong with the calculations?


![](/assets/Image(2).png)

 - Self-Attention mechanism for parallel computation, in which both input and output are the same.

 - The output results are computed at the same time, now basically has replaced the RNN

![](/assets/Image(3).png)

### Traditional word2vec
 - What's wrong with representing vectors?
 - How to represent the same word in different contexts
 - Pre-trained vectors are permanent

![](/assets/Image(4).png)

### Overall structure
 - How is the input coded?
 - What is the output result?
 - Purpose of Attention?
 - How is it put together? 

![](/assets/Image(5).png)

### What does Attention mean?
 - What is your attention for the input data?
 - How can you get the computer to pay attention to this valuable information?

![](/assets/Image(6).png)
![](/assets/Image(7).png)
 
### What is self-attention?

![](/assets/Image(8).png)
![](/assets/Image(9).png)

### How is self-attention calculated?
 - Inputs are encoded to get vectors
 - Want to get the relationship of the context of the current word, can be treated as a weighting
 - Construct three matrices to query the relationship of the current word with other words and the expression of the feature vectors respectively.

![](/assets/Image(10).png)

 - Three matrices to be trained
 - Q: query, to be queried
 - K: key, waiting to be queried
 - V: value, the actual feature information

![](/assets/Image(11).png)

 - The inner product of q and k indicates how many matches there are
 - Input two vectors to get a score
 - K: key, waiting to be queried
 - V: value, the actual feature information

![](/assets/Image(12).png)

 - The final score is softmaxed to the final contextual result.
 - Scaled Dot-Product Attention does not allow the score to increase as the vector dimension increases.

![](/assets/Image(14).png)

 - softmax recall:


![](/assets/Image(13).png)

### Attention calculation for each word
 - The Q for each word calculates a score with each K in the entire sequence and then reassigns features based on the score

![](/assets/Image(15).png)

### The overall Attention calculation process

 - Each word's Q will calculate a score with each K.
 - After Softmax, the whole weighted result is obtained.
 - At this point, each word looks at not only the sequence before it, but the entire input sequence.
 - The representation of all words is calculated at the same time.

![](/assets/Image(16).png)
![](/assets/Image(17).png)

### Multi-headed mechanisms
 - A set of q,k,v yields a set of feature expressions for the current word
 - Can a filter in a similar convolutional neural network extract multiple features?

![](/assets/Image(18).png)
 - Feature maps in convolution:
![](/assets/Image(19).png)
 - Get multiple feature representations by different HEAD
 - Stitch all features together
 - Can be dimensionalized by another layer of full concatenation

![](/assets/Image(22).png)
![](/assets/Image(23).png)
![](/assets/Image(24).png)

### Multi-headed result
 - Different attention results
 - Different feature vector representations obtained

![](/assets/Image(26).png)
![](/assets/Image(25).png)
### Stacked multilayer
 - One layer is not enough
 - The calculations are the same.

![](/assets/Image(27).png)

### Expression of positional information
In self-attention each word is weighted taking into account the whole sequence, so its occurrence position doesn't have much effect on the results, which is equivalent to it doesn't matter where it's placed, but that's a bit inconsistent with reality. We want the model to have additional knowledge about the position.

![](/assets/Image(28).png)
### Add and Normalize
 - Normalization

![](/assets/Image(29).png)
 - Connections: basic residual connections

![](/assets/Image(31).png)

![](/assets/Image(30).png)

### Decoder
 - Attention calculation is different
 - The MASK mechanism has been added

![](/assets/Image(32).png)

### Final Output Results
 - Derive the final prediction
 - The loss function cross-entropy can be

![](/assets/Image(33).png)

### Overall Sorting
 - Self-Attention
 - Multi-Head
 - Multi-layer stacking, positional encoding
 - Parallel accelerated training

![](/assets/Image(34).png)
### Effective demonstration
![](/assets/Image(35).png)

## BERT
### What is the difference between the word vectors trained by BERT?
 - In word2vec, the vectors corresponding to the same words are fixed once they are trained.
 - But in different scenarios
will 'Transformer' mean the same thing in different scenarios?
 - Both are called transformers.

![](/assets/Image(36).png)
![](/assets/Image(39).png)
### How do you explain that name?
 - Bidirectional Encoder Representations from Transformers Encoder
 - To be clear, it is the encoder part of the transformer.
 - Doesn't need labels, it can be trained with precisions.

![](/assets/Image(40).png)
### How to train BERT
 - Method 1: 15% of the words in a sentence are randomly masked off
 - Leave it to the model to predict what the masked words are.
 - There are too many possibilities for words.
 - If the BERT training vectors are good, then the classification will be OK.

![](/assets/Image(37).png)
 - Method 2: Predict whether two sentences should be joined together or not
 - [seq]: the connective before the two sentences, [cls]: the vector indicating the classification to be done

![](/assets/Image(41).png)
![](/assets/Image(38).png)

### How to use BERT
 - Isn't it necessary to train the representation of the vectors first, and then train the desired model?
 - The required tasks can be incorporated into BERT, and the two are trained together!
 - Classification of tasks:

![](/assets/Image(43).png)
![](/assets/Image(42).png)

 - For reading comprehension questions, the input is the text and the question. The output is a comprehension answer location.

![](/assets/Image(44).png) 
![](/assets/Image(45).png)

 - How to design a network? The start and end positions of the answers need to be calculated separately.

![](/assets/Image(47).png)
![](/assets/Image(46).png)

## ALBERT
### Problem to be solved (A Lite BERT)
 - Since BERT NLP has been emphasizing one thing, to be effective, the model must be large.
 - But if the model is very large, the weight parameters will be very much, the training is a big problem (video memory can not be loaded)
 - The training speed is also a thing, now the big models have to be in units of months, the speed is very slow!
 - Can we simplify BERT to make it faster and easier to train?
(Embedding in Transformer accounts for 20% of the parameters, Attetntion accounts for 80%)

### Do more hidden layer features necessarily lead to better results?

![](/assets/Image(48).png)
![](/assets/Image(49).png)

### Few letters
 - E: word embedding size, that is, the dimension of the vector obtained after the first Embedding layer.

 - H: the size of the hidden layer, e.g. 768 dimensional vectors after attentions.

 - V: the number of words in the corpus, for example, there are 20000 words in our dictionary.

### Factorization of the embedding vector parameterization

 - Through an intermediary, one layer is converted to two layers, but the number of parameters can be drastically reduced.
 - The number of parameters: (V × H) reduced to ( V × E + E × H )
 - At this point, if H >> E, we have achieved our goal (the smaller E may be less effective)
 - But Embedding layer is only the first step, Attention how to simplify is the main event!
 - The effect of different values of E on the results, smaller E affects the results, but not much.

![](/assets/Image(50).png)
### Cross-layer parameter sharing
 - There are many ways to share, ALBERT chose to share all of them, FFN and ATTENTION's all share.
![](/assets/Image(51).png)
### Stories the experiment also tells us
 - The more layers, the better. So far, yes.
![](/assets/Image(52).png)
 - The larger the hidden features, the better. So far, yes.
![](/assets/Image(53).png)

## RoBERTa
### Robustly optimized BERT approach
 - This basically means that the training process can be optimized.
 - The most important thing is how to design masks in a language model: dynamic masks are better than static ones.
 - Dynamic masks are definitely better than static ones, which is the core of this paper.
 - After canceling the NSP task (Next Sentence Prediction), the result is better.
![](/assets/Image(54).png)
### Optimal point
 - BatchSize is basically what everyone recognizes as well.
 - More datasets were used and trained for a longer period of time, improving the results a bit.
 - Split the word way to do a little improvement, so that the English split more detailed.
![](/assets/Image(55).png)
![](/assets/Image(56).png)
### RoBERTa-wwm
 - wwm is whole word mask.
 - This is quite important, 1.I like to eat XXX authentic grilled cold noodles; 2.I like to eat Haxbin authentic grilled cold noodles
 - The wwm is definitely more important for Chinese scene training.
![](/assets/Image(57).png)
## DistilBERT
### A distilled version of BERT: smaller,faster, cheaper and lighter
 - Dreaming back to 2019, back in the day, people noticed the trend of models getting bigger and bigger.
 - In academia, it's always been about violence and miracles.
 - What should we do in engineering? It has to be smaller.
 - Both small effect must be guaranteed, how to do it?
 ![](/assets/Image(58).png)
 - Almost 40% fewer parameters, mainly because of the fast prediction speed.
 - The effect remains 97% after distillation, but it's been slimmed down considerably.
![](/assets/Image(59).png)
![](/assets/Image(60).png)


## License

[MIT](https://choosealicense.com/licenses/mit/)