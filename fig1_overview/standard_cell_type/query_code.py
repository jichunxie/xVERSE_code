import pandas as pd
from openai import OpenAI

# === Load API key and initialize client ===
with open("/Users/xiaohui/Desktop/datasets/keys/openai_xielab.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# === Load input data ===
input_csv = "/Users/xiaohui/Desktop/Codes/sparest_code/standard_type/cellxgene_cell_type_id2name.csv"
df = pd.read_csv(input_csv)

# === Load prompt template ===
with open("/Users/xiaohui/Desktop/Codes/sparest_code/standard_type/prompt.txt", "r") as f:
    base_prompt = f.read().strip()

# === Define classification function ===
def classify_cell_type(name: str) -> str:
    full_prompt = f"{base_prompt}\n\nInput: {name}\n\nOutput:"
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a precise cell type classifier."},
                {"role": "user", "content": full_prompt}
            ],
        )
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return f"ERROR: {e}"

# === Apply to each cell type ===
df["classification_result"] = df["name"].apply(classify_cell_type)

# === Save output ===
output_csv = "/Users/xiaohui/Desktop/Codes/sparest_code/standard_type/cellxgene_cell_type_mapped.csv"
df.to_csv(output_csv, index=False)
print(f"Saved mapped results to: {output_csv}")