# Character Level RNN

### Imports and Libraries

~~~
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
~~~   
- NumPy (np) → Library for handling numerical data and arrays. We’ll use it to prepare and manipulate sequences of characters before sending them to the neural network.
- PyTorch (torch) → Framework for building and training neural networks. It introduces tensors, which are like NumPy arrays but can run efficiently on CPUs or GPUs.
- torch.nn (nn) → Provides ready-made layers (like RNN, LSTM, Linear) to help define models without writing the math from scratch.
- torch.nn.functional (F) → Provides functions for common operations (like activation functions, loss functions). Difference:

nn = defines the layer objects.

F = applies operations directly.

#### 1. nn (Layer Objects)

- nn is used when you want to define a layer as part of your model.
- It creates an object that has parameters (weights + biases) inside it, and these parameters are learnable (updated during training).

#### 2. F (Functional Operations)
- F is used for functions that don’t have their own parameters.
- These are mathematical operations like relu, softmax, cross_entropy, etc.
-ReLU doesn’t need to learn anything → no parameters to store.
---     
### open text file and read in data as `text`
~~~
with open('anna.txt', 'r') as f:
    text = f.read()
~~~
#### 1. with open('anna.txt', 'r') as f:

This line opens the file named anna.txt.

- 'r' means read mode → we are opening the file only to read its contents (not write).

- as f → gives the opened file the temporary name f (a file object).Using with ensures that the file will be automatically closed once we’re done (good practice in Python).

#### 2. text = f.read()

f.read() reads the entire file content into a single string.

That string is stored in the variable text.

---
### Tokenization
Neural networks don’t understand raw text (letters).
They only understand numbers (tensors).
By making this character ↔ integer mapping, we can switch back and forth:
- char2int → Encode (text → numbers).
- int2char → Decode (numbers → text).

~~~
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])
~~~
- set(text) → creates a set of unique characters in the text.
- Wrapping it in tuple(...) turns that set into an ordered collection.
- enumerate(chars) gives pairs of (index, character).   
Example: [ (0, 'h'), (1, 'e'), (2, 'l'), (3, 'o') ].

- dict(...) converts this into a dictionary:   
  {0: 'h', 1: 'e', 2: 'l', 3: 'o'}

~~~
char2int = {ch: ii for ii, ch in int2char.items()}
~~~
- This is a dictionary comprehension (shortcut for creating dictionaries).
- It flips the previous mapping, so we get char2int:   
{'h': 0, 'e': 1, 'l': 2, 'o': 3}   

Last line encodes the whole text into integers.
Now encoded is our numerical dataset, ready for the RNN.

---
### Pre-processing the data
 What is One-Hot Encoding?  
One-hot encoding is a way to represent categorical data (like characters, words, or labels) as binary vectors.      

Suppose we have 4 characters:   
 ['a', 'b', 'c', 'd'] 
- Instead of representing them as numbers (`a=0, b=1, c=2, d=3`), we create a **vector of length 4**, where only one position is `1` and the rest are `0`.

Example:  
- `a → [1, 0, 0, 0]`  
- `b → [0, 1, 0, 0]`  
- `c → [0, 0, 1, 0]`  
- `d → [0, 0, 0, 1]`

This way, the model doesn’t think that `d (3)` is somehow "larger" than `a (0)` — they are just different categories.

~~~
def one_hot_encode(arr, n_labels):

    # Initialize the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot
~~~

 one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1   
 
 - np.arange(one_hot.shape[0]) → [0, 1, 2, ...] (row indices).

- arr.flatten() → flattens arr into 1D array of class indices.

- This line sets the "correct" class index to 1 in each row.

Example: if arr = [2, 0, 1], then:   
[[0,0,1],    
 [1,0,0],     
 [0,1,0]]

 ----
### Batching

~~~
def get_batches(arr, batch_size, seq_length):   

n_batches =len(arr)//(batch_size*seq_length)  

arr =arr[:(batch_size*seq_length*n_batches)]    

arr =arr.reshape((batch_size,-1))     

for n in range(0, arr.shape[1], seq_length): 
   x =arr[:,n:n+seq_length] 
   y =np.zeros_like(x) 
   try: 
      y[:,:-1] = x[:,1:] 
      y[:, -1] = arr[:, n + seq_length] 
   except IndexError: 
      y[:,:-1] = x[:,1:] 
      y[:, -1] = arr[:, 0] 
      yield x, y
~~~
def get_batches(arr, batch_size, seq_length):
- arr: 1D array of encoded characters (numbers).
- batch_size: how many sequences (rows) we want per batch.


- seq_length: how many characters per sequence (columns).   

**1. Compute number of full batches**   
n_batches = len(arr) // (batch_size * seq_length)
- Total elements we can use = multiple of (batch_size * seq_length)
- n_batches tells how many complete batches we can make.      

 Example:
If len(arr) = 1000, batch_size = 5, seq_length = 10 →
Each batch needs 5 × 10 = 50 characters.
So n_batches = 1000 // 50 = 20.   

**2.Trim extra characters**   
arr = arr[:(batch_size * seq_length * n_batches)]    
- Keeps only the part of arr that fits perfectly into batches.
- Removes leftovers at the end that cannot form a full batch.
#### 3. Reshape into rows = batch_size
arr = arr.reshape((batch_size, -1))   
Now arr becomes a 2D matrix:   
rows = batch_size (sequences processed in parallel)    
columns = remaining characters split across time

Example:
If batch_size=2 and trimmed arr has 20 chars →   
arr.shape = (2, 10)    
Row 1: first 10 chars    
Row 2: next 10 chars

This way, each row is an independent flow of characters.

**4. Iterate over chunks of seq_length**   
for n in range(0, arr.shape[1], seq_length):  
Moves in steps of seq_length.   
Each step selects a window of length seq_length from each row. 

**5. Select features x**   
x = arr[:, n:n+seq_length]    
Takes a slice of length seq_length across all rows.   
Shape: (batch_size, seq_length)
These are the input sequences.

**6. Build targets y**   
y = np.zeros_like(x)   
Initialize y with same shape as x.    
y[:, :-1] = x[:, 1:]    
Shift x one position to the left.   
This makes y the next-character target for each input.    
Example:
If x = [7, 5, 9]
Then y = [5, 9, ?]

**7. Handle last element of sequence**   
y[:, -1] = arr[:, n + seq_length]   
For the last position in y, take the next character from the original array (continuity across chunks).   
But if n+seq_length goes out of bounds (last batch), it fails →
so we use        
except IndexError: y[:, -1] = arr[:, 0]     
to wrap around (circular continuation).

**8. Yield a batch**   
yield x, y
Returns (x, y) as one batch.

Generator continues until all batches are covered.

----

~~~
batches = get_batches(encoded, 8, 50)
x, y = next(batches)
~~~
`get_batches` is a generator.
That means instead of returning all batches at once, it yields one batch at a time when you ask for it.

So batches = get_batches(encoded, 8, 50) doesn’t give you the data directly — it gives you a `generator object` that you can loop over or fetch one batch from using next.

---
# Defining the network with PyTorch 
~~~ 
class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ## TODO: define the layers of the model
        self.lstm=nn.LSTM(len(self.chars),n_hidden,n_layers,dropout=drop_prob,batch_first=True)
        self.dropout=nn.Dropout(drop_prob)
        self.fc=nn.Linear(n_hidden,len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.reshape(-1, self.n_hidden) # Use reshape instead of view
        out = self.fc(out)
        # return the final output and the hidden state
        return out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
~~~
super().__init__() → ensures PyTorch’s own initialization happens (so your model behaves properly).

### Layers of the Model
self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)    
self.dropout = nn.Dropout(drop_prob)    
self.fc = nn.Linear(n_hidden, len(self.chars))

- nn.LSTM → the actual recurrent layer.
   - Input size = len(self.chars) (number of unique characters in your original text (the vocabulary size)).
   - Hidden size = n_hidden (number of hidden units).
   - n_layers = stacked LSTM layers.
- nn.Dropout → randomly drops some neurons to prevent overfitting.
- nn.Linear → fully connected layer: hidden → character probabilities.         
### Forward pass  
def forward(self, x, hidden):    
    r_output, hidden = self.lstm(x, hidden)   # pass data through LSTM     
    out = self.dropout(r_output)    
    out = out.reshape(-1, self.n_hidden)      # flatten output     
    for fc layer   
    out = self.fc(out)                        # map hidden →   character scores
    return out, hidden

-1 tells PyTorch: "figure this dimension out automatically".

self.n_hidden is the number of hidden units in your LSTM.

Example: if out has shape (8, 50, 256) → (batch_size=8, seq_len=50, hidden=256),
this becomes (400, 256) where 400 = 8 * 50.
Returns:   
out → predictions for next character.    
hidden → updated hidden state (so LSTM remembers context).

### hidden state initialization
weight = next(self.parameters()).data

self.parameters() gives you all the learnable parameters (weights & biases) of the model.

next(self.parameters()) just picks the first parameter tensor.

`Why?` Because this tensor already has the same device (CPU/GPU) and same datatype as the rest of your model.

.data just extracts the raw tensor values from it.

👉 So weight is basically a "template" — we don’t actually care about the values inside it.
We just need something that tells PyTorch:
"Create new tensors for hidden state that live on the same device and use the same datatype as my model."

