import torch # type: ignore
from shared.Data.RPS_ai_class import MLP # type: ignore
import os

def load_model (input_dim, hidden_dim, output_dim ):
    # Define the model
    model = MLP(input_dim, hidden_dim, output_dim)

    # Load the saved model weights into memory
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rps_model.pth")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    return model