def save_encoder_and_scaler(encoder, scaler, encoder_path='pickles/encoder.pkl', scaler_path='pickles/scaler.pkl'):
    try:
        with open(encoder_path, 'wb') as enc_file:
            pickle.dump(encoder, enc_file)
        with open(scaler_path, 'wb') as scal_file:
            pickle.dump(scaler, scal_file)
        print("Encoder and scaler saved successfully.")
    except Exception as e:
        print(f"Error saving encoder and scaler: {e}")
        

def save_data(X_scaled, y, X_scaled_path='pickles/X_scaled.pkl', y_path='pickles/y.pkl'):
    try:
        with open(X_scaled_path, 'wb') as x_file:
            pickle.dump(X_scaled, x_file)
        with open(y_path, 'wb') as y_file:
            pickle.dump(y, y_file)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")


def save_pickles(model, encoder, scaler, X_scaled, y):
    try:
        pickle_dir = 'pickles'
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
            print(f"Directory '{pickle_dir}' created.")
        else:
            print(f"Directory '{pickle_dir}' already exists.")
        
        model_path = os.path.join(pickle_dir, 'trained_model.pkl')
        
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        
        save_encoder_and_scaler(encoder, scaler)
        save_data(X_scaled, y)
        
        print(f"Model saved to '{model_path}'")
    except Exception as e:
        print(f"Error in Pickle: {e}")