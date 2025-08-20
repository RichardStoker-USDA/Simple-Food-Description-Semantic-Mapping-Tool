import streamlit as st
import pandas as pd
import numpy as np
import time
import io
from matching_functions import run_fuzzy_match, run_tfidf_match, run_embed_match
from sample_data_25 import get_sample_input_csv, get_sample_target_csv

# Page configuration
st.set_page_config(
    page_title="Simple Food Description Semantic Mapping Tool - USDA ARS",
    layout="wide",
    initial_sidebar_state="auto"  # Auto-collapse on mobile
)

# Enhanced CSS for better visibility and professional styling
st.markdown("""
<style>
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main {
            padding: 1rem 0.5rem !important;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        
        h2 {
            font-size: 1.3rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
        }
        
        .step-indicator {
            padding: 0.7rem !important;
            margin: 0.5rem 0 !important;
            font-size: 0.9rem !important;
        }
        
        .stButton > button {
            width: 100% !important;
            padding: 0.8rem !important;
        }
        
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
        }
        
        .dataframe {
            font-size: 0.75rem !important;
        }
    }
    
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Headers - softer colors */
    h1 {
        color: #2d5482;
        border-bottom: 3px solid #5a7da5;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #3d5a80;
        margin-top: 2rem;
    }
    
    h3 {
        color: #5a6c7d;
        margin-top: 1.5rem;
    }
    
    /* Step indicators - softer backgrounds */
    .step-indicator {
        background-color: #f7f9fb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #8fa3b8;
        color: #3d4852;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .step-active {
        background-color: #e8f1f9;
        border-left: 4px solid #5a8dc7;
        color: #2c4a6d;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12);
    }
    
    /* Dark mode step indicators */
    @media (prefers-color-scheme: dark) {
        .step-indicator {
            background-color: #2d3748;
            color: #e2e8f0;
            border-left-color: #718096;
        }
        
        .step-active {
            background-color: #4a5568;
            border-left-color: #63b3ed;
            color: #f7fafc;
        }
    }
    
    /* Buttons - softer colors */
    .stButton > button {
        background-color: #5a7da5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .stButton > button:hover {
        background-color: #4a6d8f;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
    }
    
    /* Sample data button - special styling */
    .sample-button > button {
        background-color: #48bb78;
        border: 2px solid #38a169;
    }
    
    .sample-button > button:hover {
        background-color: #38a169;
    }
    
    /* Success/Warning/Error messages */
    .stAlert {
        border-radius: 5px;
        padding: 1rem;
        border-left: 4px solid;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #2c5282;
    }
    
    /* Sidebar - better contrast */
    section[data-testid="stSidebar"] {
        background-color: #f7fafc;
    }
    
    /* Dark mode sidebar */
    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] {
            background-color: #1a202c;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0;
        }
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
    }
    
    /* Info boxes - softer blue */
    .stInfo {
        background-color: #f5f9fc;
        color: #3d5a80;
        border: 1px solid #b8d4e8;
    }
    
    /* Dark mode info boxes */
    @media (prefers-color-scheme: dark) {
        .stInfo {
            background-color: #2c5282;
            color: #ebf8ff;
            border-color: #63b3ed;
        }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #5a6c7d;
        font-size: 0.9em;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid #dce3e9;
    }
    
    /* Dark mode footer */
    @media (prefers-color-scheme: dark) {
        .footer {
            color: #a0aec0;
            border-top-color: #4a5568;
        }
    }
    
    /* Expander styling - softer gray */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        color: #3d4852;
    }
    
    @media (prefers-color-scheme: dark) {
        .streamlit-expanderHeader {
            background-color: #2d3748;
            color: #e2e8f0;
        }
    }
    
    /* Metrics styling - softer background */
    [data-testid="metric-container"] {
        background-color: #fafbfc;
        border: 1px solid #e8ecf0;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="metric-container"] {
            background-color: #2d3748;
            border-color: #4a5568;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'input_df' not in st.session_state:
    st.session_state.input_df = None
if 'target_df' not in st.session_state:
    st.session_state.target_df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'input_column' not in st.session_state:
    st.session_state.input_column = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'using_sample_data' not in st.session_state:
    st.session_state.using_sample_data = False

# Header
st.markdown("""
# Simple Food Description Semantic Mapping Tool
#### United States Department of Agriculture - Agricultural Research Service
""")

# Create three columns for step indicators
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.step == 1:
        st.markdown('<div class="step-indicator step-active"><strong>Step 1: Upload Files</strong><br>Upload your CSV files or try sample data</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="step-indicator">Step 1: Upload Files ✓</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.step == 2:
        st.markdown('<div class="step-indicator step-active"><strong>Step 2: Select Columns</strong><br>Choose the description columns to match</div>', unsafe_allow_html=True)
    elif st.session_state.step > 2:
        st.markdown('<div class="step-indicator">Step 2: Select Columns ✓</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="step-indicator">Step 2: Select Columns</div>', unsafe_allow_html=True)

with col3:
    if st.session_state.step == 3:
        st.markdown('<div class="step-indicator step-active"><strong>Step 3: View Results</strong><br>Review and download your matches</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="step-indicator">Step 3: View Results</div>', unsafe_allow_html=True)

st.markdown("---")

# Step 1: File Upload
if st.session_state.step == 1:
    st.header("Step 1: Upload Your Data Files")
    
    # Sample data section
    with st.container():
        st.markdown("### Try Sample Data")
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            if st.button("Load Sample Dataset (25 items)", type="secondary", use_container_width=True, 
                        help="Load a demonstration dataset with 25 food items showing various matching scores"):
                # Load sample data
                input_csv = get_sample_input_csv()
                target_csv = get_sample_target_csv()
                
                st.session_state.input_df = pd.read_csv(io.StringIO(input_csv))
                st.session_state.target_df = pd.read_csv(io.StringIO(target_csv))
                st.session_state.using_sample_data = True
                
                st.success("Sample data loaded successfully! The sample includes:")
                st.info("""
                • 3 direct matches (scores ~0.90-0.93)
                • 7 very similar items (scores ~0.88-0.92)
                • 10 moderately similar items (scores ~0.82-0.88)
                • 5 dissimilar items (scores ~0.77-0.82)
                
                Note: Semantic embedding models find relationships even in unrelated items.
                Typical scores range from 0.77-0.94. Adjust threshold to 0.85 or higher for stricter matching.
                """)
    
    st.markdown("---")
    st.markdown("### Or Upload Your Own Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input File")
        st.markdown("Upload a CSV file containing the descriptions you want to match")
        input_file = st.file_uploader(
            "Choose your input CSV file",
            type=['csv'],
            key="input_uploader",
            help="This file should contain a column with food descriptions to be matched"
        )
        
        if input_file:
            try:
                st.session_state.input_df = pd.read_csv(input_file)
                st.session_state.using_sample_data = False
                st.success(f"Loaded {len(st.session_state.input_df)} rows from input file")
                with st.expander("Preview Input Data"):
                    st.dataframe(st.session_state.input_df.head(10))
            except Exception as e:
                st.error(f"Error reading input file: {str(e)}")
        elif st.session_state.using_sample_data and st.session_state.input_df is not None:
            st.info(f"Using sample input data: {len(st.session_state.input_df)} items")
            with st.expander("Preview Sample Input Data"):
                st.dataframe(st.session_state.input_df.head(10))
    
    with col2:
        st.subheader("Target File")
        st.markdown("Upload a CSV file containing the reference descriptions to match against")
        target_file = st.file_uploader(
            "Choose your target CSV file",
            type=['csv'],
            key="target_uploader",
            help="This file should contain a column with reference food descriptions"
        )
        
        if target_file:
            try:
                st.session_state.target_df = pd.read_csv(target_file)
                st.session_state.using_sample_data = False
                st.success(f"Loaded {len(st.session_state.target_df)} rows from target file")
                with st.expander("Preview Target Data"):
                    st.dataframe(st.session_state.target_df.head(10))
            except Exception as e:
                st.error(f"Error reading target file: {str(e)}")
        elif st.session_state.using_sample_data and st.session_state.target_df is not None:
            st.info(f"Using sample target data: {len(st.session_state.target_df)} items")
            with st.expander("Preview Sample Target Data"):
                st.dataframe(st.session_state.target_df.head(10))
    
    # Navigation button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.input_df is not None and st.session_state.target_df is not None:
            if st.button("Continue to Column Selection →", type="primary", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        else:
            st.info("Please upload both input and target CSV files or load the sample dataset to continue")

# Step 2: Column Selection and Processing
elif st.session_state.step == 2:
    st.header("Step 2: Select Columns and Configure Matching")
    
    if st.session_state.using_sample_data:
        st.info("Using sample dataset - columns have been auto-selected for demonstration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Data Configuration")
        st.markdown(f"**File loaded:** {len(st.session_state.input_df)} rows")
        
        # Column selection for input
        if st.session_state.using_sample_data:
            input_column = "food_description"
            st.markdown(f"**Selected column:** {input_column} (auto-selected for sample data)")
        else:
            input_column = st.selectbox(
                "Select the column containing descriptions to match:",
                st.session_state.input_df.columns,
                key="input_col_select",
                help="Choose the column that contains the food descriptions you want to match"
            )
        st.session_state.input_column = input_column
        
        # Show sample values
        st.markdown("**Sample values from selected column:**")
        sample_inputs = st.session_state.input_df[input_column].dropna().head(5)
        for i, val in enumerate(sample_inputs, 1):
            st.text(f"{i}. {val}")
    
    with col2:
        st.subheader("Target Data Configuration")
        st.markdown(f"**File loaded:** {len(st.session_state.target_df)} rows")
        
        # Column selection for target
        if st.session_state.using_sample_data:
            target_column = "food_name"
            st.markdown(f"**Selected column:** {target_column} (auto-selected for sample data)")
        else:
            target_column = st.selectbox(
                "Select the column containing reference descriptions:",
                st.session_state.target_df.columns,
                key="target_col_select",
                help="Choose the column that contains the reference food descriptions to match against"
            )
        st.session_state.target_column = target_column
        
        # Show sample values
        st.markdown("**Sample values from selected column:**")
        sample_targets = st.session_state.target_df[target_column].dropna().head(5)
        for i, val in enumerate(sample_targets, 1):
            st.text(f"{i}. {val[:80]}..." if len(str(val)) > 80 else f"{i}. {val}")
    
    st.markdown("---")
    
    # Advanced Settings
    with st.expander("Advanced Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            methods = st.multiselect(
                "Matching Methods:",
                ["embed", "fuzzy", "tfidf"],
                default=["embed"],
                format_func=lambda x: {
                    "fuzzy": "Fuzzy String Matching",
                    "tfidf": "TF-IDF Similarity",
                    "embed": "Semantic Embeddings (Recommended)"
                }[x],
                help="Semantic Embeddings provides the highest accuracy for conceptual matching"
            )
        
        with col2:
            # Set default based on whether using sample data
            default_threshold = 0.85
            
            threshold = st.slider(
                "Similarity Threshold for NO MATCH:",
                min_value=0.0,
                max_value=1.0,
                value=default_threshold,
                step=0.05,
                help="Matches below this threshold will be marked as NO MATCH. For embedding models, try 0.85 or higher since they typically score 0.77-0.94 even for unrelated items."
            )
            st.session_state.threshold = threshold
        
        with col3:
            clean_text = st.checkbox(
                "Apply text cleaning",
                value=False,
                help="Note: Text cleaning is NOT recommended for embedding models as they work better with original text"
            )
    
    # Processing button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to File Upload", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Run Matching Process", type="primary", use_container_width=True):
            if not methods:
                st.error("Please select at least one matching method")
            else:
                # Create progress container
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Prepare data
                        status_text.text("Preparing data...")
                        progress_bar.progress(5)
                        
                        input_list = st.session_state.input_df[input_column].dropna().tolist()
                        target_list = st.session_state.target_df[target_column].dropna().tolist()
                        
                        # Remove duplicates from target list
                        target_list_unique = list(dict.fromkeys(target_list))
                        
                        status_text.text(f"Processing {len(input_list)} inputs against {len(target_list_unique)} unique targets...")
                        progress_bar.progress(10)
                        
                        # Initialize results
                        results_df = pd.DataFrame({
                            'input_description': input_list
                        })
                        
                        # Track progress
                        total_methods = len(methods)
                        progress_per_method = 80 / total_methods
                        current_progress = 10
                        
                        # Run each selected method
                        if "embed" in methods:
                            status_text.text("Loading semantic embedding model...")
                            progress_bar.progress(current_progress + 10)
                            
                            status_text.text("Computing semantic embeddings (this may take a moment)...")
                            # Don't clean text for embeddings - they work better with original text
                            embed_results = run_embed_match(input_list, target_list_unique)
                            
                            # Apply threshold
                            results_df['best_match'] = embed_results['match']
                            results_df['similarity_score'] = embed_results['score']
                            results_df.loc[results_df['similarity_score'] < threshold, 'best_match'] = 'NO MATCH'
                            
                            current_progress += progress_per_method
                            progress_bar.progress(int(current_progress))
                        
                        if "fuzzy" in methods:
                            status_text.text("Running fuzzy string matching...")
                            fuzzy_results = run_fuzzy_match(input_list, target_list_unique, clean_text)
                            results_df['fuzzy_match'] = fuzzy_results['match']
                            results_df['fuzzy_score'] = [s/100.0 for s in fuzzy_results['score']]  # Normalize to 0-1
                            
                            # Apply threshold
                            results_df.loc[results_df['fuzzy_score'] < threshold, 'fuzzy_match'] = 'NO MATCH'
                            
                            current_progress += progress_per_method
                            progress_bar.progress(int(current_progress))
                        
                        if "tfidf" in methods:
                            status_text.text("Running TF-IDF matching...")
                            tfidf_results = run_tfidf_match(input_list, target_list_unique, clean_text)
                            results_df['tfidf_match'] = tfidf_results['match']
                            results_df['tfidf_score'] = tfidf_results['score']
                            
                            # Apply threshold
                            results_df.loc[results_df['tfidf_score'] < threshold, 'tfidf_match'] = 'NO MATCH'
                            
                            current_progress += progress_per_method
                            progress_bar.progress(int(current_progress))
                        
                        # Round scores for display
                        for col in results_df.columns:
                            if 'score' in col:
                                results_df[col] = results_df[col].round(4)
                        
                        status_text.text("Finalizing results...")
                        progress_bar.progress(95)
                        
                        st.session_state.results = results_df
                        
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        time.sleep(1)
                        st.session_state.step = 3
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

# Step 3: Results
elif st.session_state.step == 3:
    st.header("Step 3: Matching Results")
    
    if st.session_state.using_sample_data:
        st.info("Sample Data Results - Score distribution ranges from near-perfect matches (0.936) to weak matches (0.775). With threshold 0.85: 19 matches, 6 NO MATCH items")
    
    if st.session_state.results is not None:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_inputs = len(st.session_state.results)
        
        with col1:
            st.metric("Total Inputs", total_inputs)
        
        if 'best_match' in st.session_state.results.columns:
            no_matches = (st.session_state.results['best_match'] == 'NO MATCH').sum()
            with col2:
                st.metric("Successful Matches", total_inputs - no_matches)
            with col3:
                st.metric("No Matches", no_matches)
            with col4:
                avg_score = st.session_state.results[st.session_state.results['best_match'] != 'NO MATCH']['similarity_score'].mean()
                st.metric("Avg Match Score", f"{avg_score:.3f}" if not pd.isna(avg_score) else "N/A")
        
        st.markdown("---")
        
        # Results table
        st.subheader("Detailed Results")
        
        # Add filtering options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_term = st.text_input("Search/filter results:", placeholder="Type to filter...")
        with col2:
            # Check if there are any NO MATCH items to show
            has_no_matches = False
            if 'best_match' in st.session_state.results.columns:
                has_no_matches = (st.session_state.results['best_match'] == 'NO MATCH').any()
            
            if has_no_matches:
                show_only_no_match = st.checkbox("Show only NO MATCH items")
            else:
                show_only_no_match = False
                st.checkbox("Show only NO MATCH items (adjust threshold to create NO MATCH items)", 
                           value=False, disabled=True)
        with col3:
            sort_by_score = st.checkbox("Sort by similarity score", value=True)
        
        # Filter and sort results
        display_df = st.session_state.results.copy()
        
        if search_term:
            mask = display_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            display_df = display_df[mask]
        
        if show_only_no_match and 'best_match' in display_df.columns:
            display_df = display_df[display_df['best_match'] == 'NO MATCH']
        
        if sort_by_score and 'similarity_score' in display_df.columns:
            display_df = display_df.sort_values('similarity_score', ascending=False)
        
        # Display with styling
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Score distribution if using sample data
        if st.session_state.using_sample_data and 'similarity_score' in st.session_state.results.columns:
            st.markdown("---")
            st.subheader("Score Distribution (Sample Data)")
            scores = st.session_state.results['similarity_score']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                excellent = (scores >= 0.92).sum()
                st.metric("Excellent (≥0.92)", excellent)
            with col2:
                very_good = ((scores >= 0.88) & (scores < 0.92)).sum()
                st.metric("Very Good (0.88-0.92)", very_good)
            with col3:
                good = ((scores >= 0.85) & (scores < 0.88)).sum()
                st.metric("Good (0.85-0.88)", good)
            with col4:
                moderate = ((scores >= 0.82) & (scores < 0.85)).sum()
                st.metric("Moderate (0.82-0.85)", moderate)
            with col5:
                weak = (scores < 0.82).sum()
                st.metric("Weak (<0.82)", weak)
            
            st.caption("Note: Semantic models typically score 0.77-0.94. Even unrelated items share some semantic space, which is why scores rarely drop below 0.75.")
        
        # Download section
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Back to Column Selection", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        
        with col2:
            csv = st.session_state.results.to_csv(index=False)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="Download Complete Results (CSV)",
                data=csv,
                file_name=f"semantic_matching_results_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
        
        with col3:
            if st.button("Start New Analysis →", use_container_width=True):
                # Reset everything
                st.session_state.step = 1
                st.session_state.results = None
                st.session_state.input_df = None
                st.session_state.target_df = None
                st.session_state.using_sample_data = False
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>Western Human Nutrition Research Center</strong> | Davis, CA<br>
    Diet, Microbiome and Immunity Research Unit<br>
    United States Department of Agriculture | Agricultural Research Service<br>
    <br>
    <small>Based on <a href='https://github.com/mike-str/USDA-Food-Description-Mapping' target='_blank'>USDA Food Description Mapping Research</a></small>
</div>
""", unsafe_allow_html=True)