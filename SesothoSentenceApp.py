import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random
import base64

# Set up page configuration
st.set_page_config(page_title="Sesotho NLP Tool", page_icon="", layout="centered")

# Paths to Sesotho Models
MODEL_DIR = "sesotho_model"
TOKENIZER_DIR = "sesotho_tokenizer"

# Common Sesotho starter words)
SESOTHO_SEEDS = [
    "Ke", "Mohlankana","Sebaka", "Batho", "Metsi", "Lefatše", "Thuto", "Morena", 
    "Ha", "Re", "O", "Lekhotla", "Basotho", "Mosali", "Monna", "Bana", "Lesotho", "Sechaba", "Tona Kholo", "Papali", "A", "Sebele", "Ngoana" 
]

# 1. Load local model and tokenizer with caching
@st.cache_resource
def load_local_nlp_assets(model_path, tokenizer_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder missing at: '{model_path}'")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer folder missing at: '{tokenizer_path}'")
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_local_nlp_assets(MODEL_DIR, TOKENIZER_DIR)
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Initialization Error: {e}")
    model_loaded = False

#st.title("Sesotho Next-Word & Sentence Generator")
import streamlit as st

st.markdown(
    """
    <style>
    .navy-header-container {
        background-color: #FFFFFF; /* Deep navy blue background */
        color: #0A192F;             /* Light text for high contrast */
        padding: 10px 5px;
        border-radius: 1px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        max-width: 850px;
        margin: 20px auto;
    }
    
    /* Flexbox wrapper for the logo and the main titles */
    .header-top-row {
        display: flex;
        align-items: center;        /* Vertically centers the logo with the two lines */
        gap: 24px;                  /* Generous space between logo and text */
        margin-bottom: 15px;        /* Beautiful gap before the description paragraph */
    }
    
    .header-logo {
        width: 90px;                /* Explicit size for a crisp look */
        height: auto;
        object-fit: contain;
    }
    
    .header-text-column {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .navy-header-container h1 {
        font-size: 36px;
        font-weight: 700;
        color: #0A192F;
        margin: 0 0 6px 0;          /* Tight gap between main title and subtitle */
        line-height: 1.1;
        letter-spacing: 1px;
        padding-bottom: 5px;
    }
    
    .navy-header-container h2 {
        font-size: 20px;
        font-weight: 400;
        color: #0D20BF;            /* Soft slate-blue secondary text */
        margin: 0;
        line-height: 1.3;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding-top: 2px;
    }
    
    .navy-header-container p {
        font-size: 18px;
        font-weight: 300;
        color: #0D20BF;            /* Readability optimized body text */
        margin: 0;
        line-height: 1.6;          /* Perfectly spaced paragraph lines */
        border-top: 2px solid #172A45;
        padding-top: 5px;         /* Creates a clean separating line before the description */
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    with open("mologo.png", "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode()
    img_src = f"data:image/png;base64,{base64_string}"
except FileNotFoundError:
    # Safe fallback link if the image is missing from the folder
    img_src = "AppLogo"

st.markdown(
    f"""
    <div class="navy-header-container">
        <div class="header-top-row">
            <img src="{img_src}" class="header-logo" alt="Logo">
            <div class="header-text-column">
                <h1>MOPHETHELO - MATHETHO</h1>
                <h2>Sesotho next-word & sentence generator</h2>
            </div>
        </div>
        <p>Leqhepe lena le u lumella ho sebelisa AI ho etsa polelo kapa ho fumana lentsoe le hlahlamang.</p>
    </div>
    """,
    unsafe_allow_html=True
)


#Etsa 2 columns
col1, col2 = st.columns(2)

with col1:
    custom_css = """
    <style>
        .st-key-custom_container {
            border: 4px solid #0A192F !important; /* Changes size to 4px and color to blue */
            border-radius: 8px;               /* Optional: rounds the corners */
            padding: 15px;                      /* Optional: adds breathing room inside */
        }
    </style>
    """
    st.html(custom_css)

    with st.container(border=True, key="custom_container"):
        st.header("Mphe lentsoe le hlahlamang")
        #st.subheader("Mphe lentsoe le hlahlamang")
        user_input = st.text_input("Thaepa lentsoe kapa mantsoe a qalang:", "Morena", key="word_pred_input")
        num_predictions = st.slider("Mantsoe a latelang la hau:", min_value=1, max_value=5, value=3)

        if model_loaded and user_input.strip():
            with st.spinner("Rea inahana..."):
                inputs = tokenizer(user_input, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = outputs.logits
                
                next_token_logits = predictions[0, -1, :]
                probabilities = torch.softmax(next_token_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probabilities, num_predictions)
                
                suggestions = []
                for prob, idx in zip(top_k_probs, top_k_indices):
                    word = tokenizer.decode([idx]).strip()
                    if word:
                        suggestions.append((word, prob.item() * 100))

            if suggestions:
                for idx, (word, confidence) in enumerate(suggestions, 1):
                    st.write(f"**{idx}. {word}** — *{confidence:.2f}% monyetla*")
                    st.progress(confidence / 100)
            else:
                st.warning("Re na le bothata ba ho fumana lentsoe le hlahlamang.")
        elif not model_loaded:
            st.warning("Application idle: Model failed to load.")

with col2:
    custom_css = """
    <style>
        .st-key-custom_container2 {
            border: 4px solid #0A192F  !important; /* Changes size to 4px and color to green */
            border-radius: 8px;               /* Optional: rounds the corners */
            padding: 15px;                      /* Optional: adds breathing room inside */
        }
    </style>
    """
    st.html(custom_css)
    with st.container(border=True, key="custom_container2"):
        st.header("Mphe Polelo")
        #st.subheader("Mphe Polelo")
        st.write("Penya konopo ea Mphe Polelo, ha u batla ho etsa polelo e qetellang ka khutlo kapa potso.")
        
        max_length = st.slider("Nomoro ea mantsoe a etsang polelo (tokens):", min_value=5, max_value=50, value=25)
        temperature = st.slider("Khakanyo (Temperature):", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

        if st.button("🎲 Mphe Polelo"):
            if model_loaded:
                with st.spinner("Re ntse re u lokisetsa polelo..."):
                    seed_word = random.choice(SESOTHO_SEEDS)
                    input_ids = tokenizer.encode(seed_word, return_tensors="pt")
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            max_length=max_length,
                            do_sample=True,
                            temperature=temperature,
                            top_k=50,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    generated_text = " ".join(generated_text.split()).strip()
                    
                    # --- UPDATED POST-PROCESSING LOGIC (SUPPORT PERIOD & QUESTION MARK) ---
                    # Find the last occurrence index of either a period or a question mark
                    last_period = generated_text.rfind(".")
                    last_question = generated_text.rfind("?")
                    
                    # Determine which valid punctuation comes last in the string
                    last_punctuation_index = max(last_period, last_question)
                    
                    if last_punctuation_index != -1:
                        # Cut off everything after the last valid punctuation mark
                        generated_text = generated_text[:last_punctuation_index + 1]
                    else:
                        # If neither was generated, strip trailing symbols and default to a period
                        generated_text = generated_text.rstrip(",;:-?! ")
                        generated_text += "."
                    
                    # Display results beautifully
                    st.success("### Polelo e ncha:")
                    st.info(f"🌱 **Lentsoe le qalang:** {seed_word}")
                    st.write(f"📝 *\"{generated_text}\"*")
            else:
                st.error("Model is not loaded. Cannot generate text.")

