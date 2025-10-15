import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage
from tensorflow.keras.models import load_model
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Molecular Solubility Predictor",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🧪 Molecular Solubility Predictor")
st.write(
    "Enter a SMILES string of a molecule to predict its aqueous solubility (logS). "
    "This app uses three different machine learning models to provide a comprehensive prediction."
)

# --- Model and Featurization Functions ---

@st.cache_resource
def load_models():
    """
    Loads the trained models.
    Uses caching to prevent reloading on every interaction.
    """
    try:
        rf_model = joblib.load('medical/random_forest_model.pkl')
        xgb_model = joblib.load('medical/xgboost_model.pkl')
        nn_model = load_model('medical/neural_network_model.keras')
        return rf_model, xgb_model, nn_model
    except FileNotFoundError:
        st.error(
            "Model files not found. Please make sure 'random_forest_model.pkl', "
            "'xgboost_model.pkl', and 'neural_network_model.keras' "
            "are in the same directory as this app.py file."
        )
        return None, None, None

def featurize_smiles(smiles):
    """
    Converts a SMILES string to a Morgan Fingerprint.
    Uses the same parameters as the training script.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # IMPORTANT: Use the exact same fingerprint parameters from your training
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fingerprint)

def generate_molecule_image(smiles):
    """
    Generates a PIL Image of the molecule's 2D structure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        pil_img = MolsToGridImage([mol], molsPerRow=1, subImgSize=(400, 400), useSVG=False)
        return pil_img
    return None

# --- Main Application Logic ---

# Load the models once
rf_model, xgb_model, nn_model = load_models()

# User input
st.subheader("Enter Molecule Information")
user_smiles = st.text_input(
    "SMILES String:",
    "CC(=O)Oc1ccccc1C(=O)O", # Default example: Aspirin
    help="Enter the SMILES notation of the molecule you want to analyze."
)

if st.button("Predict Solubility", type="primary"):
    if not user_smiles:
        st.warning("Please enter a SMILES string.")
    elif rf_model is None:
        # Error message is already displayed by load_models()
        pass
    else:
        # --- 1. Generate and display molecule image ---
        st.subheader("Molecule Structure")
        mol_image = generate_molecule_image(user_smiles)
        if mol_image:
            st.image(mol_image, use_container_width=True)
        else:
            st.error(f"Invalid SMILES string provided: {user_smiles}")

        # --- 2. Featurize and Predict ---
        fingerprint = featurize_smiles(user_smiles)
        if fingerprint is not None:
            input_data = fingerprint.reshape(1, -1)

            # Get predictions from all models
            pred_rf = rf_model.predict(input_data)[0]
            pred_xgb = xgb_model.predict(input_data)[0]
            pred_nn = nn_model.predict(input_data).flatten()[0]

            st.subheader("Predicted Solubility (logS)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Neural Network", f"{pred_nn:.4f}", "Best Model")
            col2.metric("XGBoost", f"{pred_xgb:.4f}")
            col3.metric("Random Forest", f"{pred_rf:.4f}")
        else:
            # This case is redundant if the image fails, but good for safety
            st.error("Could not generate features from the provided SMILES string.")