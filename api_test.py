# import stanza
#
# nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
# doc = nlp("Robert I. Toussie, general partner of the investment group, said the Lionel response reflected management's entrenched position, saying officials had failed to come up with a better alternative to his group's offer. Mr. Toussie said he would respond to Lionel's suit after his lawyers review it.")
#
# print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


# from transformers import pipeline
#
# classifier = pipeline("fill-mask")
# res = classifier("Paris is the <mask> of France.")
#
# print(res)


from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")

sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

my_head_mask = torch.ones([12, 12], dtype=torch.float64)
for i in range(3, 12):
    my_head_mask[10][i] = 0
print(input)
token_logits = model(input, head_mask=my_head_mask)[0]
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))