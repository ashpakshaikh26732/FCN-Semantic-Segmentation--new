def get_images_and_segments_test_arrays():

  y_true_segments = []
  y_true_images = []
  test_count = 64

  ds = validation_dataset.unbatch()
  ds = ds.batch(101)

  for image, annotation in ds.take(1):
    y_true_images = image
    y_true_segments = annotation


  y_true_segments = y_true_segments[:test_count, : ,: , :]
  y_true_segments = np.argmax(y_true_segments, axis=3)

  return y_true_images, y_true_segments


y_true_images, y_true_segments = get_images_and_segments_test_arrays()


results = model.predict(validation_dataset, steps=validation_steps)


results = np.argmax(results, axis=3)