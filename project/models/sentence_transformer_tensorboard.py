import os
from typing import Iterable, Optional, Union

from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class SentenceTransformerTensorboard(SentenceTransformer):
    def __init__(
		self, 
        writer: SummaryWriter,
        model_name_or_path: Optional[str] = None,
		modules: Optional[Iterable[nn.Module]] = None,
		device: Optional[str] = None,
		cache_folder: Optional[str] = None,
		use_auth_token: Union[bool, str, None] = None,
	):
        super().__init__(model_name_or_path, modules, device, cache_folder, use_auth_token)
        self.writer = writer

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
		# super._eval_during_training(evaluator, output_path, save_best_model, epoch, steps, callback)
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
		
        self.writer.add_scalar('Validation/Accuracy', score, epoch)
