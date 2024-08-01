import tensorflow as tf

def check_tensorflow_chips():
  chip_types = ['GPU', 'CPU', 'TPU']
  for chip in chip_types:
    devices = tf.config.list_physical_devices(chip)
    if devices:
      print(f"{chip}s available: {len(devices)}")
      for device in devices:
        print(device.name)
    else:
      print(f"None found for {chip}")

def check_gpu_memory():
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus:
      details = tf.config.experimental.get_device_details(gpu)
      print(f"GPU Details: {details}")
    try:
      with tf.device('/GPU:0'):
        a = tf.random.uniform([1024, 1024])
        b = tf.random.uniform([1024, 1024])
        c = tf.matmul(a, b)
      print("Simple TensorFlow operation executed successfully on GPU.")
    except Exception as e:
      print(f"Error during TensorFlow operation: {e}")
  else:
    print("No GPU devices found.")

if __name__ == "__main__":
  check_tensorflow_chips()
  check_gpu_memory()
