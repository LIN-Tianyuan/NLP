# NLP

## Natural Language Processing BERT Model

### Transformer
#### What's the one thing to do?
 - The basic composition is still the Seq2Seq network, which is common in machine translation modeling.

 - The inputs and outputs are intuitive, and the core architecture is the network design in the middle.


![](/assets/Image(1).png)

#### Traditional RNN networks
 - What's wrong with the calculations?


![](/assets/Image(2).png)

 - Self-Attention mechanism for parallel computation, in which both input and output are the same.

 - The output results are computed at the same time, now basically has replaced the RNN

![](/assets/Image(3).png)

#### Traditional word2vec
 - What's wrong with representing vectors?
 - How to represent the same word in different contexts
 - Pre-trained vectors are permanent

![](/assets/Image(4).png)

#### Overall structure
 - How is the input coded?
 - What is the output result?
 - Purpose of Attention?
 - How is it put together? 

![](/assets/Image(5).png)

#### What does Attention mean?
 - What is your attention for the input data?
 - How can you get the computer to pay attention to this valuable information?

![](/assets/Image(6).png)
![](/assets/Image(7).png)
 
#### What is self-attention?

![](/assets/Image(8).png)
![](/assets/Image(9).png)

#### How is self-attention calculated?
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

#### Attention calculation for each word
 - The Q for each word calculates a score with each K in the entire sequence and then reassigns features based on the score

![](/assets/Image(15).png)

#### The overall Attention calculation process

 - Each word's Q will calculate a score with each K.
 - After Softmax, the whole weighted result is obtained.
 - At this point, each word looks at not only the sequence before it, but the entire input sequence.
 - The representation of all words is calculated at the same time.

![](/assets/Image(16).png)
![](/assets/Image(17).png)

#### Multi-headed mechanisms
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

#### Multi-headed result
 - Different attention results
 - Different feature vector representations obtained

![](/assets/Image(26).png)
![](/assets/Image(25).png)
#### Stacked multilayer
 - One layer is not enough
 - The calculations are the same.

![](/assets/Image(27).png)

#### Expression of positional information
In self-attention each word is weighted taking into account the whole sequence, so its occurrence position doesn't have much effect on the results, which is equivalent to it doesn't matter where it's placed, but that's a bit inconsistent with reality. We want the model to have additional knowledge about the position.

![](/assets/Image(28).png)
#### Add and Normalize
 - Normalization

![](/assets/Image(29).png)
 - Connections: basic residual connections

![](/assets/Image(31).png)

![](/assets/Image(30).png)

#### Decoder
 - Attention calculation is different
 - The MASK mechanism has been added

![](/assets/Image(32).png)

#### Final Output Results
 - Derive the final prediction
 - The loss function cross-entropy can be

![](/assets/Image(33).png)

#### Overall Sorting
 - Self-Attention
 - Multi-Head
 - Multi-layer stacking, positional encoding
 - Parallel accelerated training

![](/assets/Image(34).png)
#### Effective demonstration
![](/assets/Image(35).png)

### BERT
#### What is the difference between the word vectors trained by BERT?
 - In word2vec, the vectors corresponding to the same words are fixed once they are trained.
 - But in different scenarios
will 'Transformer' mean the same thing in different scenarios?
 - Both are called transformers.

![](/assets/Image(36).png)
![](/assets/Image(39).png)
#### How do you explain that name?
 - Bidirectional Encoder Representations from Transformers Encoder
 - To be clear, it is the encoder part of the transformer.
 - Doesn't need labels, it can be trained with precisions.

![](/assets/Image(40).png)
#### How to train BERT
 - Method 1: 15% of the words in a sentence are randomly masked off
 - Leave it to the model to predict what the masked words are.
 - There are too many possibilities for words.
 - If the BERT training vectors are good, then the classification will be OK.

![](/assets/Image(37).png)
 - Method 2: Predict whether two sentences should be joined together or not
 - [seq]: the connective before the two sentences, [cls]: the vector indicating the classification to be done

![](/assets/Image(41).png)
![](/assets/Image(38).png)

#### How to use BERT
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
## License

[MIT](https://choosealicense.com/licenses/mit/)