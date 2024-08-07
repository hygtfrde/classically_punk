import os
import importlib.util

import tensorflow as tf

def get_user_input(prompt, default_value=True):
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'n']:
            return response == 'y'
        elif response == '':
            return default_value
        else:
            print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")


def prompt_for_gpu():
    response = input("Do you want to use GPU for training if available? (Y/N): ").strip().lower()
    if response == 'y':
        script_path = os.path.join('scripts', 'config_tf.py')
        if os.path.exists(script_path):
            spec = importlib.util.spec_from_file_location("config_tf", script_path)
            config_tf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_tf)
            print("GPU setup completed successfully.")
            print('To manage GPU usage run: \n tensorboard --logdir=logs/fit')
        else:
            print("GPU configuration script not found.")
    else:
        print("Using CPU defaults for training.")
        tf.config.set_visible_devices([], 'GPU')