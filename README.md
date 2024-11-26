# llm-etf-prediction

## ✅ How to run
### 1) Prerequisites
To run this model, the following prerequisites are required:
- CUDA Version: 11.8
- Nvidia Driver Version: 470.103.01

```bash
conda create -n etf python=3.10.0
conda activate etf
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
pip install torch==2.5.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy==2.1.3
pip install neuralforecast
pip install transformers
pip install matplotlib
pip install ipywidgets
pip install yfinance
```

### 2) Clone
Clone this repository to your local computer using:
```bash
https://github.com/JeonHaeseung/llm-etf-prediction.git
```

### 3) Run
In the `time-llm.py` file, you can select your desired ETF to create the model.
```python
python3 time-llm.py
```

To check the result graph, you can run `result.py`.
```python
python3 result.py
```

## 📊 Dataset
Stock price datasets can be retrieved via API calls using Python's `yfinance` library. Natural language description data for the LLM is hard-coded in `time-llm.py`. If you want to fetch descriptions for a new stock, search for the stock ticker on `https://money.usnews.com` and use the description provided.