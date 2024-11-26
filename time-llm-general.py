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
        "SPY": "The dataset contains historical closing prices of the SPY (SPDR S&P 500 ETF Trust). \
                The SPDR S&P 500 ETF Trust is an exchange-traded fund which trades on the NYSE Arca under the symbol SPY. \
                The ETF is designed to track the S&P 500 index by holding a portfolio comprising all 500 companies on the index. \
                The S&P 500 is widely regarded as the best gauge of overall performance in large-capitalized US equities, \
                and is comprised of 500 American companies representing a wide range of diverse market sectors.",
        "SMH": "The dataset contains historical closing prices of the SMH (VanEck Semiconductor ETF). \
                The VanEck Semiconductor ETF intended to track the overall performance of companies involved in semiconductor production and equipment. \
                The investment seeks to replicate as closely as possible, before fees and expenses, the price and yield performance of the MVIS® US Listed Semiconductor 25 Index. \
                The fund normally invests at least 80 percent of its total assets in securities that comprise the fund's benchmark index. \
                The index includes common stocks and depositary receipts of U.S. exchange-listed companies in the semiconductor industry.",
        "IBB": "The dataset contains historical closing prices of the IBB (iShares Biotechnology ETF). \
                The investment seeks to track the investment results of the NYSE Biotechnology Index composed of U.S.-listed equities in the biotechnology sector. \
                The fund generally will invest at least 80 percent of its assets in the component securities of its index and in investments \
                that have economic characteristics that are substantially identical to the component securities of its index and may invest up to 20 percent of its assets in certain futures, options and swap contracts, cash and cash equivalents.",
        "GDX": "The dataset contains historical closing prices of the GDX (VanEck Gold Miners ETF). \
                The investment seeks to replicate as closely as possible, before fees and expenses, the price and yield performance of the NYSE® Arca Gold Miners Index®. \
                The fund normally invests at least 80 percent of its total assets in common stocks and depositary receipts of companies involved in the gold mining industry.\
                The index is a modified market-capitalization weighted index primarily comprised of publicly traded companies involved in the mining for gold and silver.",
        "IYR": "The dataset contains historical closing prices of the IYR (iShares US Real Estate ETF). \
                The investment seeks to track the investment results of the Dow Jones U.S. Real Estate Capped Index. \
                The fund seeks to track the investment results of the Dow Jones U.S. Real Estate Capped Index, which measures the performance of the real estate sector of the U.S. equity market, as defined by the index provider. \
                It generally invests at least 80 percent of its assets in the component securities of its underlying index and in investments that have economic characteristics that are substantially identical to the component securities of its underlying index.",
        "KBE": "The dataset contains historical closing prices of the KBE (SPDR® S&P Bank ETF). \
                The investment seeks to provide investment results that, before fees and expenses, correspond generally to the total return performance of the S&P Banks Select Industry Index. \
                The fund generally invests substantially all, but at least 80%, of its total assets in the securities comprising the index. \
                The index represents the banks segment of the S&P Total Market Index (“S&P TMI”). The S&P TMI is designed to track the broad U.S. equity market. \
                It may invest in equity securities that are not included in the index, cash and cash equivalents or money market instruments, such as repurchase agreements and money market funds."
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