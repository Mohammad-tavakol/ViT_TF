import tensorflow as tf
from tensorflow.keras import layers

"""TRansformer Encoder >>> norms, Multi-Head Attention, and mlp layer"""
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, n_heads, mlp_dim):
        super(TransformerEncoder, self).__init__(name="transformer_encoder")
        self.norm1 = layers.LayerNormalization()#
        self.norm2 = layers.LayerNormalization()
        self.attention = layers.MultiHeadAttention(num_heads=n_heads, key_dim= embed_dim) #this embed dimension here is required because output size should be equal to input from patch embedding
        self.mlp = tf.keras.Sequential([layers.Dense(mlp_dim,activation=tf.nn.gelu),
                                        layers.Dense(embed_dim,activation=tf.nn.gelu),
                                        ])
    
    def call(self,input):
        x = self.norm1(input)
        x = self.attention(x,x)

        #building the residual branch inside the transformer
        x = tf.math.add(x,input)

        x_1 = self.norm2(x)
        output = self.mlp(x_1)
        output=tf.math.add(output,x)
        return output