from bert_score import score
from rouge import Rouge

def get_metrics(
    candidates: list,  # ["xxxx"]
    refs: list,        # ["xxxx"]
    lang: str = "en",  # Options: [en, zh]
    ):
    
    P, R, F1 = score(
        cands=candidates,
        refs=refs,
        lang=lang
    )

    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidates, [ref for ref in refs], avg=True)
    rouge_l = rouge_scores['rouge-l']['f']

    # print(f"ROUGE-L F1: {rouge_l:.4f}")
    return {
        "Precision": P.item(), 
        "Recall": R.item(), 
        "F1_Score": F1.item(),
        "ROUGE_L": rouge_l,
        }

if __name__ == "__main__":
    candidates = ["the cat sits on mat"]
    refs=["a cat is on the mat"]
    res = get_metrics(candidates, refs, lang="en")
    print(res)