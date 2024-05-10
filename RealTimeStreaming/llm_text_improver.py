from llama_cpp import Llama
import numpy as np

THRESHOLD = 512
llm = Llama(model_path="model_paths/llama-2-7b.Q3_K_S.gguf")

prompt = """You are going to be given a sentence which may be malformed. 
Given the conversation context, return the sentence with corrected grammar and makes the most sense:
"""


def get_next_sentence(sentance, llm, prompt):
    sentances = []
    token_count = []
    sent_count = 0
    while True:
        sent_count += 1
        sentances.append(f"sentence {sent_count}: " + sentance)
        token_count.append(len(sentance)/4)

        if np.sum(token_count) > THRESHOLD:
            token_count = token_count[1:]
            token_count.pop()
            sentances.pop()

        prompt += '\n'.join(sentances)
        prompt += f"\nupdated sentence {sent_count}: "
        chosen_sentance = llm(prompt, stop=["\n"])["choices"][0]["text"]
        sentances[-1] = f"sentence {sent_count}: " + chosen_sentance
        sentance = yield chosen_sentance


gen = get_next_sentence("What a nice day is!", llm, prompt)

sentance_1 = gen.__next__()
print(sentance_1)
sentance_2 = gen.send("Should we beach go?")
print(sentance_2)
sentance_3 = gen.send("Yes I want to go to the.")
print(sentance_3)



