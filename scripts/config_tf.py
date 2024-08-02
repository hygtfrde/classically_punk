import tensorflow as tf

def configure_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            for gpu in gpus:
                tf.config.set_memory_growth(gpu, True)
                
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU Details: {details}")
                
            print("GPU setup completed successfully.")
        except RuntimeError as e:
            print(f"Error during GPU setup: {e}")
    else:
        print("No GPU devices found. Using CPU.")

if __name__ == "__main__":
    configure_gpus()
