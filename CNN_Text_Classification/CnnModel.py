
#Loading data
#('x: ', array([[32, 22, 17, 12, 33,  7, 32,  2,  8,  0, 20, 10,  5, 31, 16,  0, 14,
#        33, 19,  4, 27, 13, 15, 30,  6, 23,  1, 18,  9, 34, 11, 21, 28, 24],
#       [26,  1, 25,  5, 29,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
#         3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3]]))
#('y: ', array([[0, 1],
#       [1, 0]]))
#('vocabulary: ', {'and': 5, 'the': 32, 'century': 8, 'splash': 27, 'is': 17, 'conan': 10, 'steven': 28, 'schwarzenegger': 23, 'even': 13, 'segal': 24, '21st': 2, 'van': 34, 'make': 19, ',': 1, 'to': 33, 'going': 14, 'new': 20, 'be': 7, 'tedious': 29, "'s": 0, 'greater': 15, 'that': 31, '<PAD/>': 3, 'simplistic': 26, 'destined': 12, 'damme': 11, 'claud': 9, 'than': 30, 'he': 16, 'a': 4, 'arnold': 6, 'silly': 25, 'rock': 22, 'jean': 18, 'or': 21})
#('vocabulary_inv: ', ["'s", ',', '21st', '<PAD/>', 'a', 'and', 'arnold', 'be', 'century', 'claud', 'conan', 'damme', 'destined', 'even', 'going', 'greater', 'he', 'is', 'jean', 'make', 'new', 'or', 'rock', 'schwarzenegger', 'segal', 'silly', 'simplistic', 'splash', 'steven', 'tedious', 'than', 'that', 'the', 'to', 'van'])


from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from sklearn.cross_validation import train_test_split
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from data_helpers import load_data
from keras.optimizers import Adam
from keras.models import Model

print 'Loading data'
x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1]

##print("sequence_length: ", sequence_length);
#print("x: ", x);
#print("y: ", y);
#print("vocabulary: ", vocabulary);
#print("vocabulary_inv: ", vocabulary_inv);

vocabulary_size = len(vocabulary_inv)
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

nb_epoch = 100
batch_size = 30

# this returns a tensor
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
flatten = Flatten()(merged_tensor)
# reshape = Reshape((3*num_filters,))(merged_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(output_dim=2, activation='softmax')(dropout)

# this creates a model that includes
model = Model(input=inputs, output=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training




