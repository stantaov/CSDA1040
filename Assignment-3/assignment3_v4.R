library(tcltk) 
library(keras)
library(EBImage)
library(tensorflow)



# list of categories
face_categories <- c("happy", "sad") 

# number of output classes
output_n <- length(face_categories)

img_width <- 244
img_height <- 244
target_size <- c(img_width, img_height)
channels <- 1

# path to image folders
train_image_files_path <- tk_choose.dir()
valid_image_files_path <- tk_choose.dir()

train_data_gen = image_data_generator(
  rescale = 1/255)

valid_data_gen <- image_data_generator(
  rescale = 1/255
)  


# loading traning images
train_images <- flow_images_from_directory(train_image_files_path, 
                                           train_data_gen,
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           classes = face_categories,
                                           color_mode = "grayscale",
                                           seed = 42)
# loading test images
valid_images <- flow_images_from_directory(valid_image_files_path, 
                                           valid_data_gen,
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           classes = face_categories,
                                           color_mode = "grayscale",
                                           seed = 42)


table(factor(train_images$classes))


train_images$class_indices


classes_indices <- train_images$class_indices
#save(classes_indices, file = "/Users/stantaov/Documents/face.RData")


# number of training samples
train_samples <- train_images$n

# number of validation samples
valid_samples <- valid_images$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10


# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(200) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "SGD",
  metrics = "accuracy"
)


# fit
hist <- model %>% fit_generator(
  # training data
  train_images,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_images,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 1,
  callbacks = list(
    # save best model after every epoch
    #callback_model_checkpoint("/Users/stantaov/Documents/keras/face.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    #callback_tensorboard(log_dir = "/Users/stantaov/Documents/keras/logs")
  )
)

tensorboard("/Users/stantaov/Documents/keras/logs")

#####################

plot(hist)

#####################


#BiocManager::install("rhdf5")
#library("rhdf5")
#model_final <- H5Fopen("/Users/stantaov/Documents/keras/face.h5")
#h5ls(model_final)
#str(model_final)



path <- tk_choose.files()
img <- readImage(path)
display(img)
img <- resize(img, w = 244, h = 244)
display(img)
x <- image_to_array(img)
x <- array_reshape(x, c(1, dim(x)))
preds <- model %>% predict(x)

preds
