import pandas as pd
from tqdm import tqdm
from transformers import LlamaTokenizer

# https://github.com/yangjianxin1/LLMPruner
# 如果我们重新从头开始训练，那么使用起来比较容易，但是如果我们是把两个模型进行合并的，在初始化的时候想让原来的token使用pretrain阶段的结果，而没有的token重新初始化，可以参考上面的项目做法。

def cal_lens():
    tokenizer = LlamaTokenizer.from_pretrained("merged_tokenizer_hf_40k")
    llama_tokenizer = LlamaTokenizer.from_pretrained("llama")

    print(tokenizer)

    num_tokens = []
    num_ids = []
    num_ids_llama = []
    with open("data/corpus_csl.txt", "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            encode = tokenizer(line)
            encode_llama = llama_tokenizer(line)
            # print(encode)
            # print(len(line),len(encode['input_ids']))
            num_tokens.append(len(line))
            num_ids.append(len(encode["input_ids"]))
            num_ids_llama.append(len(encode_llama["input_ids"]))

    df = pd.DataFrame(
        {"num_tokens": num_tokens, "num_ids": num_ids, "num_ids_llama": num_ids_llama}
    )
    df = df.sort_values(by=["num_tokens"], ascending=True)
    df.to_csv("data/merged_tokenizer_hf_40k_efficiency.csv", index=False)


def eda_lens():
    data = pd.read_csv("data/merged_tokenizer_hf_40k_efficiency.csv")
    print(data.columns)
    data.columns = ["num_tokens_raw", "num_ids_gogpt", "num_ids_llama"]
    print(data.describe(percentiles=[0.3, 0.9]))

    with pd.ExcelWriter(r"resources/tokenizer_efficiency_eda.xlsx") as writer:
        data[data["num_tokens_raw"] <= 100].reset_index(drop=True).to_excel(
            writer, sheet_name="num_tokens_raw<=100"
        )
        data[
            (data["num_tokens_raw"] > 100) & (data["num_tokens_raw"] <= 500)
        ].reset_index(drop=True).to_excel(writer, sheet_name="num_tokens_raw_100_500")
        data[
            (data["num_tokens_raw"] > 500) & (data["num_tokens_raw"] <= 1000)
        ].reset_index(drop=True).to_excel(writer, sheet_name="num_tokens_raw_500_1000")
        data[
            (data["num_tokens_raw"] > 1000) & (data["num_tokens_raw"] <= 2000)
        ].reset_index(drop=True).to_excel(writer, sheet_name="num_tokens_raw_1000_2000")
        data[data["num_tokens_raw"] > 2000].reset_index(drop=True).to_excel(
            writer, sheet_name="num_tokens_raw_2000"
        )


def eda_stats():
    data = pd.read_csv("data/merged_tokenizer_hf_40k_efficiency.csv")
    print(data.columns)
    data.columns = ["num_tokens_raw", "num_ids_gogpt", "num_ids_llama"]
    data["raw_gogpt_ratio"] = data["num_tokens_raw"] / data["num_ids_gogpt"]
    data["llama_gogpt_ratio"] = data["num_ids_llama"] / data["num_ids_gogpt"]
    print(data.describe(percentiles=[0.3, 0.9]))
    data.describe(percentiles=[0.3, 0.9]).to_excel(
        "resources/eda_stats.xlsx", index=True
    )


if __name__ == "__main__":
    # eda_lens()
    eda_stats()
