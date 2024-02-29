# NLP

## BERT

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
## License

[MIT](https://choosealicense.com/licenses/mit/)