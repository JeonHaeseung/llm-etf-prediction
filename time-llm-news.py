import os
import time
import numpy as np
import logging
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
ticker = "KBE"
log_file = os.path.join(log_dir, f'{ticker}.txt')

gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
gpt2 = GPT2Model.from_pretrained('openai-community/gpt2',config=gpt2_config)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')


def setup_logging():
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def log_message(message, level='info'):
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    else:
        logging.debug(message)


def load_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    close_prices = stock_data[['Close']].reset_index()
    close_prices = close_prices.rename(columns={"Date": "ds", "Close": "y"})
    close_prices['ds'] = pd.to_datetime(close_prices['ds'])
    close_prices = close_prices.dropna()
    return close_prices


def load_prompt():
    prompt_dict = {
        "SMH": "The dataset contains historical closing prices of the SMH (VanEck Semiconductor ETF). \
                The VanEck Semiconductor ETF intended to track the overall performance of companies involved in semiconductor production and equipment. \
                The investment seeks to replicate as closely as possible, before fees and expenses, the price and yield performance of the MVIS® US Listed Semiconductor 25 Index. \
                The fund normally invests at least 80 percent of its total assets in securities that comprise the fund's benchmark index. \
                The index includes common stocks and depositary receipts of U.S. exchange-listed companies in the semiconductor industry.",
    }

    prompt_prefix = prompt_dict[ticker]
    return prompt_prefix


def fill_missing(df, start_date, end_date):
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.set_index('ds').reindex(full_date_range).reset_index()
    df.columns = ['ds', 'y']
    df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
    return df


def prepare_data():
    train_start_date = "2019-01-01"
    train_end_date = "2024-06-01"

    test_start_date = "2024-06-02"
    test_end_date = "2024-11-25"

    train_df = load_stock_data(ticker, train_start_date, train_end_date)
    test_df = load_stock_data(ticker, test_start_date, test_end_date)

    train_df = fill_missing(train_df, train_start_date, train_end_date)
    test_df = fill_missing(test_df, test_start_date, test_end_date)

    # len(train_df):  1979 len(test_df):  177

    train_df['unique_id'] = ticker
    test_df['unique_id'] = ticker

    return train_df, test_df


def train_model(train_df, test_df, prompt_prefix):

    timellm = TimeLLM(h=7,                          # 예측하고자 하는 미래 시계열의 기간(2주일)
                    input_size=90,                  # 과거 데이터를 입력으로 사용할 시점의 수 (3개월)
                    llm=gpt2,
                    llm_config=gpt2_config,
                    llm_tokenizer=gpt2_tokenizer,
                    prompt_prefix=prompt_prefix,    # 입력 시퀀스에 추가되는 프롬프트(문맥 정보)
                    max_steps=100,
                    batch_size=128,
                    windows_batch_size=128,          # 시계열 데이터의 슬라이딩 윈도우를 사용하여 처리하는 배치 크기
                    n_heads=4,
                    early_stop_patience_steps=5,
                    num_workers_loader=64
                    )

    nf = NeuralForecast(models=[timellm], freq='D')

    print("Traning the model...")
    nf.fit(df=train_df, val_size=12, verbose=True)

    print("Save the model...")
    nf.save(path=f'./checkpoints/{ticker}/',
        model_index=None, 
        overwrite=True,
        save_dataset=True)


def load_and_predict_model(test_df):
    nf = NeuralForecast.load(path=f'./checkpoints/{ticker}/')
    forecasts = nf.predict(futr_df=test_df).reset_index()

    log_message("test_df:")
    log_message(test_df.head())
    log_message(test_df.tail())

    log_message("forecasts:")
    log_message(forecasts.head())
    log_message(forecasts.tail())

    return forecasts


def save_csv(forecasts, csv_path):
    df = pd.DataFrame(forecasts)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    setup_logging()
    train_df, test_df = prepare_data()
    prompt_prefix = load_prompt()
    train_model(train_df, test_df, prompt_prefix)
    forecasts = load_and_predict_model(test_df)
    save_csv(forecasts, csv_path=f"./result/{ticker}.csv")