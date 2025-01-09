"""ViT class definition"""
import tensorflow as tf
from tensorflow.keras import layers,Model
from patch_embedding import PatchEmbedding
from transformer_encoder import TransformerEncoder

class ViTClassifier(Model):
    def __init__(self,embed_dim, n_heads, att_mlp_size, patch_size, image_size,n_att_layers, num_classes, mlp_classifier_size):
        super(ViTClassifier, self).__init__(name="ViT-classifier")
        """args:
        embed_dim: the dimension of embedding>> can be defined arbitrarily>> yet commonly is considered to be equal to mlp_size
        n_heads: nubmber of heads for the multi head attention block
        att_mlp_size: should be (patch_size**2)*channels
        patch_size: integer 
        image_size: heigh and width of image>>should be equal 
        n_att_layers: number of stacked attention encoders
        mlp_classifier_size: number of neurons in first layer of Dense classifier
        """
        self.n_att_layers=n_att_layers
        num_patches=(image_size//patch_size)**2
        self.patch_embedding=PatchEmbedding( patch_size=patch_size,embed_dim=embed_dim,image_size=image_size)
        self.trans_encoders=[TransformerEncoder(embed_dim=embed_dim,n_heads=n_heads,mlp_dim=att_mlp_size) for _ in range(n_att_layers)]
        self.mlp_classifier=tf.keras.Sequential([layers.Dense(mlp_classifier_size,activation=tf.nn.gelu),
                                                 layers.Dense(mlp_classifier_size//2,activation=tf.nn.gelu), #making network smaller
                                                 layers.Dense(num_classes,activation="softmax")])#last layer with nerons equal to number of classes
    
    def call(self, input, training=True):
        x = self.patch_embedding(input)
        for i in range(self.n_att_layers): #the input passes through all stacked transformer layers
            x = self.trans_encoders[i](x)
        
        #now the transformer structure is ended and we want to pass the data to the MLP classifier 
        x = layers.Flatten()(x)  
        output = self.mlp_classifier(x) 
        return output
