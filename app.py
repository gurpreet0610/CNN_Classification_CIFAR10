import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import csv



(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

input_file = open("Data/accuracy.csv","r+")
reader_file = csv.reader(input_file)
value = len(list(reader_file)) + 1

path="Data/models/model"+str(value)+".h5"

model.save(path)

with open("Data/accuracy.csv", "a+") as fp:
    wr = csv.writer(fp)
    wr.writerow([value,test_acc,path])







