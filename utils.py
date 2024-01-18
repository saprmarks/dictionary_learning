from datasets import load_dataset

def hf_dataset_to_generator(dataset_name, split='train', streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    def gen():
        for x in iter(dataset):
            yield x['text']
    
    return gen()