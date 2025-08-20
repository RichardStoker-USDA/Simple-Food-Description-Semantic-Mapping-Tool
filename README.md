# Simple Food Description Semantic Mapping Tool

Match food descriptions to reference databases using advanced NLP. Built by USDA ARS for nutrition research. Works instantly online or runs locally.

## Key Features

- **High-Accuracy Semantic Matching**: Uses GTE-large embedding model (1024-dimensional vectors)
- **Multiple Algorithms**: Semantic embeddings, fuzzy string matching, and TF-IDF
- **Sample Dataset**: 25 pre-loaded food items for instant demonstration
- **NO MATCH Detection**: Configurable threshold for identifying poor matches
- **Real-time Processing**: Progress tracking and status updates

## Quick Start

### Try Online Demo
**Live Demo:** [https://simple-food-description-semantic-mapping-tool.streamlit.app/](https://simple-food-description-semantic-mapping-tool.streamlit.app/)

Click the link above to instantly try the app - includes sample data for testing!

### Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Simple-Food-Description-Mapping-Tool.git
cd Simple-Food-Description-Mapping-Tool
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open in browser:**
Navigate to http://localhost:8501

## Usage Guide

### Quick Demo with Sample Data

1. Click **"Load Sample Dataset (25 items)"** button
2. Columns are auto-selected for you
3. Click **"Run Matching Process"**
4. View results with similarity scores ranging from 0.77-0.94

### Using Your Own Data

#### Prepare Your Files
- **Input CSV**: Food descriptions to match (e.g., "apple juice", "chicken breast")
- **Target CSV**: Reference descriptions (e.g., "Apple juice, unsweetened, bottled")

#### Process Your Data
1. Upload both CSV files
2. Select the description columns
3. Adjust similarity threshold (default: 0.85)
4. Run matching process
5. Download results as CSV

## Understanding Similarity Scores

The GTE-large semantic embedding model produces nuanced similarity scores:

| Score Range | Interpretation | Example |
|------------|---------------|---------|
| 0.92-1.0 | Excellent match | "brown rice cooked" → "Rice, brown, long-grain, cooked" (0.936) |
| 0.90-0.92 | Very good match | "apple juice" → "Apple juice, unsweetened, bottled" (0.913) |
| 0.88-0.90 | Good match | "chicken breast grilled" → "Chicken, breast, grilled" (0.904) |
| 0.85-0.88 | Moderate match | "pasta with tomato sauce" → "Pasta, cooked, enriched" (0.863) |
| < 0.85 | Weak match/NO MATCH | "xyz123 test item" → NO MATCH (0.783) |

**Sample Data Results**: With the 25-item demo dataset and 0.85 threshold:
- **19 successful matches** (76% match rate)
- **6 NO MATCH items** (including nonsense text like "xyz123 test item", "synthetic compound ABC")
- **Average match score**: 0.897 for successful matches
- Score distribution ranges from ~0.775 (poor matches) to 0.936 (near-perfect matches)

## Technical Specifications

### Models & Algorithms

| Method | Model/Library | Use Case |
|--------|--------------|----------|
| Semantic Embeddings | thenlper/gte-large | Best for conceptual similarity |
| Fuzzy Matching | RapidFuzz (Levenshtein) | Good for typos and variations |
| TF-IDF | Scikit-learn | Effective for keyword matching |

### System Requirements
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended for large datasets)
- ~1.5GB disk space for model downloads

### Dependencies
- streamlit (≥1.48.1)
- pandas (≥2.3.1)
- numpy (≥2.3.2)
- scikit-learn (≥1.7.1)
- sentence-transformers (≥5.1.0)
- rapidfuzz (≥3.13.0)
- torch (≥2.8.0)

## File Structure

```
Simple-Food-Description-Mapping-Tool/
├── app.py                 # Main application with responsive design
├── matching_functions.py  # Core matching algorithms
├── sample_data_25.py     # Sample dataset generator
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore          # Git exclusions
```

## About

**Developed by:**  
Diet, Microbiome and Immunity Research Unit  
Western Human Nutrition Research Center  
USDA Agricultural Research Service  
Davis, California

**Based on research from:** [USDA Food Description Mapping](https://github.com/mike-str/USDA-Food-Description-Mapping)  
*Contributors list pending*

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact Richard Stoker
- Email: richard.stoker@usda.gov

---

*Last updated: August 2025*
