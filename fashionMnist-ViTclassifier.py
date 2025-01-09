import tensorflow as tf
from vit_classifier import ViTClassifier



#load dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()


# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print(x_train.shape)
x_train = x_train[..., tf.newaxis]  # Add channel dimension
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
print(x_test.shape)


# Define parameters for model and dataset
patch_size=7
channels=1
n_heads=1
embed_dim=(patch_size**2)*channels
att_mlp_size=(patch_size**2)*channels
image_size=28
n_att_layers=1
num_classes=10
mlp_classifier_size=16
batch_size=128

# Instantiate the ViT model
vit = ViTClassifier(
    embed_dim=embed_dim,
    n_heads=n_heads,
    att_mlp_size=att_mlp_size,
    patch_size=patch_size,
    image_size=image_size,
    n_att_layers=n_att_layers,
    num_classes=num_classes,
    mlp_classifier_size=mlp_classifier_size
)

# Compile the model
vit.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

#test inference
test=vit(tf.zeros([batch_size,image_size,image_size,1]))
print(test.shape)
vit.summary()


# Train the model
history= vit.fit(x_train, y_train, batch_size=batch_size, epochs=30, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt
#plot loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","val"])

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","val"])
plt.tight_layout()
plt.show()