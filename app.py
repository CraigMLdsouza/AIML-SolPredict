import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import MolsToGridImage
from tensorflow.keras.models import load_model
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="SolubilityX AI Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Modern UI Styling ---
st.markdown("""
<style>
    /* General background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }

    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Titles */
    h1, h2, h3 {
        color: #58a6ff;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Sidebar headings and text */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p {
        color: #e6edf3 !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.75rem;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2ea043, #3fb950);
        transform: scale(1.02);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #79c0ff;
        font-weight: 700;
        font-size: 2em;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1em;
    }


    /* Scale label */
    .sol-scale {
        font-size: 0.9em;
        color: #c9d1d9;
    }
    
    /* Styling for st.container with border */
    [data-testid="stVerticalBlock"] {
        border: 1px solid #30363d !important;
        background-color: #161b22 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("💊 SolubilityX AI Predictor")
st.markdown("""
Welcome to **SolubilityX**, an advanced AI tool for predicting aqueous solubility (**logS**).  
Instantly screen potential drug candidates by providing a SMILES string and get a comprehensive analysis from leading machine learning models.
""")
st.markdown("---")

# --- Model and Helper Functions ---
@st.cache_resource
def load_models():
    """Loads trained models from files with caching."""
    try:
        models = {
            'Random Forest': joblib.load('medical/random_forest_model.pkl'),
            'XGBoost': joblib.load('medical/xgboost_model.pkl'),
            'Neural Network': load_model('medical/neural_network_model.keras')
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}. Ensure models are in the 'medical' subfolder.")
        return None

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fingerprint), mol

def generate_molecule_image(mol):
    if mol:
        return MolsToGridImage([mol], molsPerRow=1, subImgSize=(400, 400), useSVG=False)
    return None

def create_solubility_scale(logs_value):
    min_logS, max_logS = -12, 2
    norm_val = max(0, min(1, (logs_value - min_logS) / (max_logS - min_logS)))
    red, green = int(255 * (1 - norm_val)), int(255 * norm_val)
    color = f'rgb({red}, {green}, 0)'

    scale_html = f"""
    <div class="sol-scale">Insoluble <span style="float:right;">Very Soluble</span></div>
    <div style="background: linear-gradient(to right, #ff4b4b, #ffcc00, #4caf50); border-radius: 5px; height: 10px; width: 100%; position: relative; margin-top: 5px;">
        <div style="position: absolute; left: {norm_val*100}%; top: -5px; width: 4px; height: 20px; background-color: {color}; border: 2px solid white; border-radius: 2px; transform: translateX(-50%); box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>
    </div>
    """
    return scale_html

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls & Information")

    example_molecules = {
        "Select an example": "",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Hypothetical Molecule": "Clc1ccc(cc1)S(=O)(=O)N1CCOCC1",
    }

    selected_example = st.selectbox("💡 Load Example Molecule", list(example_molecules.keys()))

    st.subheader("Select Models to Run")
    models_to_run = {
        'Neural Network': st.checkbox('Neural Network', value=True),
        'XGBoost': st.checkbox('XGBoost', value=True),
        'Random Forest': st.checkbox('Random Forest', value=True),
    }

    st.markdown("---")
    st.info("""
    **About logS:** LogS represents the logarithm of a compound’s solubility in water.  
    More negative values → lower solubility (poor drug absorption).
    """)

# --- Main App Logic ---
all_models = load_models()
user_smiles = st.text_input(
    "🧪 Enter SMILES String or Select an Example",
    value=example_molecules[selected_example],
    help="Type a SMILES string here or choose an example molecule from the sidebar."
)

if st.button("🚀 Predict Solubility", use_container_width=True):
    active_models = [name for name, active in models_to_run.items() if active]

    if not user_smiles:
        st.warning("Please enter a SMILES string to start analysis.")
    elif not all_models:
        st.error("Models could not be loaded. Please check your files.")
    elif not active_models:
        st.warning("Please select at least one model to run.")
    else:
        fingerprint, mol = featurize_smiles(user_smiles)

        if fingerprint is not None and mol is not None:
            input_data = fingerprint.reshape(1, -1)
            predictions = {}

            for model_name in active_models:
                model = all_models[model_name]
                pred = model.predict(input_data).flatten()[0] if model_name == 'Neural Network' else model.predict(input_data)[0]
                predictions[model_name] = pred

            best_model = min(predictions, key=lambda k: abs(predictions[k])) if predictions else None

            st.markdown("---")
            st.header("📈 Prediction Results")

            cols = st.columns(len(active_models) or 1)
            for i, model_name in enumerate(active_models):
                with cols[i]:
                    with st.container(border=True):
                        delta = "⭐ Best Prediction" if model_name == best_model else None
                        st.metric(
                            label=f"**{model_name}**",
                            value=f"{predictions[model_name]:.4f}",
                            delta=delta,
                            delta_color="off" if delta is None else "inverse"
                        )
                        st.markdown(create_solubility_scale(predictions[model_name]), unsafe_allow_html=True)
            
            st.markdown("---")
            st.header("🔬 Molecule Analysis")

            with st.container(border=True):
                col_img, col_props = st.columns([1.2, 1])
                with col_img:
                    st.subheader("2D Structure")
                    st.image(generate_molecule_image(mol), use_container_width=True)
                with col_props:
                    st.subheader("Physicochemical Properties")
                    st.json({
                        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
                        "LogP (Lipophilicity)": f"{Descriptors.MolLogP(mol):.2f}",
                        "H-Bond Donors": Descriptors.NumHDonors(mol),
                        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
                        "TPSA": f"{Descriptors.TPSA(mol):.2f} Å²"
                    })
        else:
            st.error(f"Invalid SMILES string: `{user_smiles}`")

