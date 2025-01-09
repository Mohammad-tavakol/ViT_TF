import tensorflow as tf
from tensorflow.keras import layers


"""Patch and positional Embedding"""
#patcht tokenization >>flatten patches>>linear projection of patches
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, image_size):
        super(PatchEmbedding,self).__init__(name="patch_encoder")
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.image_size = image_size
        self.projection=layers.Dense(embed_dim)
        self.positional_embedding = layers.Embedding(input_dim= (image_size // patch_size)**2, output_dim=embed_dim) #input_dim= num_patches, output_dim=embed_dim
        
    
    def call(self,images):
        #patch extraction
        patches = tf.image.extract_patches(images = images, sizes = [1, self.patch_size, self.patch_size, 1],
                                           strides = [1,self.patch_size, self.patch_size, 1],
                                           rates = [1, 1, 1, 1],
                                           padding="VALID")
        #output patches have a dimension of [batch_size, image_height//patch_size, image_width//patch_sizepatch_size, num_patches]
        
        #Flatten patches with reshape function
        patch_dims = patches.shape[-1]  #number of arrays of patches
        patches = tf.reshape(patches, [tf.shape(patches)[0], int(self.image_size/self.patch_size)**2, patch_dims])  # (batch_size, num_patches, patch_size^2 * channels)
        patch_embeddings = self.projection(patches)

        #adding positional embeddings
        num_patches=tf.shape(patches)[1]
        positions=tf.range(start=0,limit=num_patches, delta=1)
        position_embeddings =  self.positional_embedding(positions)

        embeddings=patch_embeddings+position_embeddings

        return embeddings
    