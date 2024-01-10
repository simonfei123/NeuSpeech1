from torchmetrics.text.bert import BERTScore
import datasets

import evaluate

bertscore = BERTScore()


def compute_metrics(preds, labels):


    scores = bertscore(preds, labels)
    scores={key:scores[key].mean()*100 for key in scores.keys()}
    return scores


class FullEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description='None',
            citation='None',
            inputs_description='None',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[""],
            reference_urls=[
                "",
            ],
        )
    def _compute(self, predictions, references):

        return compute_metrics(predictions,references)