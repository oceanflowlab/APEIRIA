import libs.capeval.bleu.bleu as capblue
import libs.capeval.cider.cider as capcider
import libs.capeval.rouge.rouge as caprouge
import libs.capeval.meteor.meteor as capmeteor
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

def score_captions(corpus: dict, candidates: dict):
    """
    adapted from Vote2Cap-DETR
    """

    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    try:
        meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    except Exception as e:
        logger.warning("METEOR failed:")
        print(e)
        meteor = deepcopy(rouge)


    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }

    message = "\n".join(
        [
            "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
            ),
            "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
            ),
            "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
            ),
            "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
            ),
            "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                cider[0], max(cider[1]), min(cider[1])
            ),
            "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                rouge[0], max(rouge[1]), min(rouge[1])
            ),
            "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                meteor[0], max(meteor[1]), min(meteor[1])
            ),
        ]
    )

    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric