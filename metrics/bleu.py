from torchmetrics import Metric
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import RougeScorer


class BLEURougeScore(Metric):

    def __init__(self):
        self.bleu_calculator = CorpusBleuCalculator()
        self.rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def update(self, predictions, references):
        self.bleu_calculator.add(predictions, references)
        self.rouge_scorer.update(predictions, references)

    def compute(self):
        bleu_score = self.bleu_calculator.score()
        rouge_scores = self.rouge_scorer.score()
        return {
            'bleu': bleu_score,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL']
        }