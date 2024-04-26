***
# Section 1
For the task of embedding H3 hexagons, both Doc2Vec and Word2Vec can be adapted. Here's a comparison of both approaches:

Word2Vec:

    Token-Level Embedding: Word2Vec is designed for token-level embeddings, where each token (word) is embedded into a continuous vector space.
    No Context: Word2Vec does not inherently capture the context of tokens. It treats each token as an independent unit.
    Requires Contextual Information: If you have additional contextual information associated with each hexagon (such as nearby hexagons or attributes), Word2Vec might be less suitable because it doesn't directly incorporate this context.

Doc2Vec:

    Document-Level Embedding: Doc2Vec is designed for embedding entire documents or sequences of tokens, treating each document as a single unit.
    Contextual Information: Doc2Vec inherently captures the context of tokens within a document, which could be useful if you have additional contextual information associated with each hexagon.
    Scalability: Doc2Vec might be less scalable compared to Word2Vec for large datasets, as it needs to process entire documents.

In the case of H3 hexagons, where each hexagon can be considered as a standalone entity without much inherent context, Word2Vec might be more straightforward and efficient. However, if you have additional contextual information associated with each hexagon (such as spatial relationships or attributes), Doc2Vec might be more suitable as it can capture this context.

Ultimately, the choice between Word2Vec and Doc2Vec depends on the specific characteristics of your dataset and the type of information you want to capture in the embeddings. It's a good idea to experiment with both approaches and evaluate their performance based on your task requirements.

***
# Section 2

To adapt Word2Vec for embedding H3 hexagons, we can treat each hexagon as a "word" and create a corpus of hexagon sequences. We'll use the Gensim library for Word2Vec implementation. Here's an example:
    
```python
from gensim.models import Word2Vec

# List of H3 hexagons
hexagons = ["88283082bffffff", "882830828ffffff", "88283082affffff"]

# Split hexagons into lists of characters (treat each hexagon as a "word")
hexagon_sequences = [[char for char in hexagon] for hexagon in hexagons]

# Train Word2Vec model
model = Word2Vec(sentences=hexagon_sequences, vector_size=100, window=5, min_count=1, workers=4)

# Get the embeddings for the hexagons
embeddings = {hexagon: model.wv[hexagon] for hexagon in hexagons}

print("Embeddings:")
for hexagon, embedding in embeddings.items():
    print(hexagon, ":", embedding)
```
    
In this code:

- We split each H3 hexagon into a list of characters, treating each hexagon as a "word".
- We train a Word2Vec model on the hexagon sequences.
- Finally, we obtain the embeddings for each hexagon from the trained Word2Vec model.

This approach adapts Word2Vec to the task of embedding H3 hexagons by treating them as discrete units of information. However, it's important to note that Word2Vec may not fully capture the spatial relationships or semantics inherent in H3 hexagons as well as techniques specifically designed for geographical data.


***
# Section 3



Using Word2Vec or Doc2Vec directly for H3 hexagons might not be ideal, as these algorithms are designed for natural language processing tasks and require sequences of tokens as input. However, we can adapt the concept by treating each hexagon as a "word" and embedding it into a continuous vector space.

Here's how you could use Doc2Vec from the Gensim library to embed H3 hexagons:

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# List of H3 hexagons
hexagons = ["88283082bffffff", "882830828ffffff", "88283082affffff"]

# Tag the hexagons with unique identifiers
tagged_data = [TaggedDocument(words=[hexagon], tags=[str(i)]) for i, hexagon in enumerate(hexagons)]

# Train Doc2Vec model
model = Doc2Vec(vector_size=100, min_count=1, epochs=10)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Get the embeddings for the hexagons
embeddings = [model.infer_vector([hexagon]) for hexagon in hexagons]

print("Embeddings:")
for hexagon, embedding in zip(hexagons, embeddings):
    print(hexagon, ":", embedding)
```

In this code:

- Each H3 hexagon is treated as a "document" with a unique identifier.
- We train a Doc2Vec model on the tagged hexagons.
- Finally, we infer the embedding for each hexagon using the trained model.

Note that the quality of the embeddings might vary depending on factors such as the size of the dataset and the similarity of the hexagons.
You may need to experiment with the hyperparameters of the Doc2Vec model for optimal results. Additionally, keep in mind that Doc2Vec is designed for text data and may not capture the spatial relationships inherent in H3 hexagons as well as other methods specifically tailored for geographical data.

***
# Section 4
import tensorflow as tf
from tensorflow.keras import layers, models

```python
class HexagonTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(HexagonTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [self.encoder_layer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model)

        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def encoder_layer(self, d_model, num_heads, dff, rate=0.1):
        return tf.keras.Sequential([
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
            layers.Dropout(rate),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(dff, activation='relu'),
            layers.Dropout(rate),
            layers.Dense(d_model),
            layers.Dropout(rate),
            layers.LayerNormalization(epsilon=1e-6)
        ])
```
***
