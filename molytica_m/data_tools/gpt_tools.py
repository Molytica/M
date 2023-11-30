from openai import OpenAI

def ask_gpt(input_text, prompt, model, temperature):
    with open("molytica_m/data_tools/.env") as file:
        api_key = file.read()
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key,
    )

    gpt_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model=model,
        temperature=temperature,
        timeout=10,
    )
    return gpt_response.choices[0].message.content