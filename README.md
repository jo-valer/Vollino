# Vollino 🏐

This is my HMD 2025 course project.\
Vollino is an intelligent conversational agent that assists Trentino Volley fans with team-related services including ticket purchases, merchandise orders, match schedules, news updates, and match results.


## ✨ Features

### 🎫 Supported Intents
- **Buy Tickets** - Purchase tickets for male/female teams with various sectors and options
- **Buy Merchandise** - Order team merchandise (shirts, scarfs, hats, balls) with size/quantity selection
- **Match Schedule** - Get upcoming match dates and times
- **News Updates** - Latest news for male, female, or youth teams
- **Match Results** - Recent game results and scores
- **Information Requests** - Ask questions about tickets or merchandise

### 🌍 Multilingual Support
- **English** - Native language support
- **Italian** - Full translation support using DeepL API

### 🎨 User Interface Options
- **GUI Mode** - Graphical interface
- **Terminal Mode** - Command-line interaction for development/testing


## 🚀 Installation & Setup

### Prerequisites
- Python 3.13
- Conda

### 1. Clone the Repository
```bash
git clone https://github.com/jo-valer/Vollino.git
cd Vollino
```

### 2. Create Python Environment
```bash
conda create --name vollino python=3.13
conda activate vollino
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file based on `.env.example` and configure your API keys:
```bash
DEEPL_API_KEY=your_deepl_api_key_here
HF_ACCESS_TOKEN=your_hf_access_token_here
```

## 🎮 Usage

### Basic Usage with GUI
```bash
python pipeline.py llama3 --interface
```

### Italian Language Mode
```bash
python pipeline.py llama3 --interface --avoid-input-translation --user-id 1
```


## 📊 Evaluation

Sbatch files for evaluation:
- **`evaluate.sh`** - Evaluate all components
- **`evaluate_translation.sh`** - Evaluate explicit translation NLU
- **`evaluate_implicit.sh`** - Evaluate implicit multilingual processing NLU


## 🗃️ Data Sources

### Static Data
- **`data/prices.json`** - Ticket and merchandise pricing
- **`data/results_*.json`** - Historical match results
- **`data/user_settings.json`** - User preferences and history

### Dynamic Data (Web Scraping)
- **Trentino Volley Official Website** - Live news and match updates
- Automatic data refresh for current information


# 🛠️ Project Structure
```
Vollino/
├── components/         # Core system components
│   ├── nlu.py          # Natural Language Understanding
│   ├── dst.py          # Dialogue State Tracking
│   ├── dm.py           # Dialogue Management
│   ├── nlg.py          # Natural Language Generation
│   ├── translator.py   # Translation utilities
│   ├── prompts.py      # LLM prompts and templates
│   └── utils.py        # Shared utilities
├── data/               # Data files and management
│   ├── data.py         # Data access functions
│   └── *.json          # Static data files
├── evaluation/         # Evaluation scripts and metrics
├── gui_interface.py    # GUI implementation
├── pipeline.py         # Main system orchestrator
└── requirements.txt    # Python dependencies
```

