from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

class TextDerivator():
    def __init__(self, language, device):
        self.language = language
        self.device = device
    
    def derive(self, all_text):
        lens = [len(text.split(' ')) for text in all_text]
        lens.sort()
        print("ORIGINAL: Min length: "+str(lens[0])+" max length: "+str(lens[-1])+" median: "+str(lens[int(0.5*len(lens ))])+" 1%: "+str(lens[int(0.01*len(lens ))]))
        set_seed(10)
        shortened = [' '.join(text.split(' ')[:10]) for text in all_text]
        if self.language == 'en':
            generator = pipeline('text-generation', model='gpt2', framework = "pt", device = self.device)
        elif self.language == 'es':
            model_id = "PlanTL-GOB-ES/gpt2-base-bne"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            generator = pipeline('text-generation', tokenizer=tokenizer, model=model, framework = "pt", device = self.device)
        BATCH_SIZE = 16
        result = []
        batches = [shortened[i:i + BATCH_SIZE] for i in range(0, len(shortened), BATCH_SIZE)]
        progress_bar = tqdm(range(len(batches)), ascii=True)
        for batch in batches:
            outs = generator(batch, max_length=100, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
            result_here = [x[0]['generated_text'] for x in outs]
            result.extend(result_here)
            progress_bar.update(1)
        lens = [len(text.split(' ')) for text in result]
        lens.sort()
        print("DERIVED: Min length: " + str(lens[0]) + " max length: " + str(lens[-1]) + " median: " + str(
        lens[int(0.5 * len(lens))]) + " 1%: " + str(lens[int(0.01 * len(lens))]))
        return result