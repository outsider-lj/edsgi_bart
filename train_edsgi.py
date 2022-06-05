# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import json
import logging
import math
import os
from collections import defaultdict
from itertools import chain
from pprint import pformat

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from torch.nn.parallel import DistributedDataParallel
from bart import BartTokenizer, AdamW, BartConfig,StepBartForDialogueGeneration#,shift_tokens_right
from config import Config
from utils_getdataloader import get_data_loaders
import random
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
EMO_PAD_TOKEN_ID=8
# MAX_LENGTH = [148,14, 30, 38]
n = ['NN','NNP','NNPS','NNS','UH']#5
v = ['VB','VBD','VBG','VBN','VBP','VBZ']#6
a = ['JJ','JJR','JJS']#3
r = ['RB','RBR','RBS','RP','WRB']#5
EMO_LABELS =["<joyful>","<excited>","<proud>","<grateful>","<hopeful>","<content>","<prepared>","<anticipating>","<confident>",
                "<sentimental>","<nostalgic>","<trusting>","<faithful>","<caring>","<terrified>","<afraid>", "<anxious>","<apprehensive>",
                "<lonely>","<embarrassed>","<ashamed>","<guilty>","<sad>","<disappointed>","<devastated>",
                "<angry>","<annoyed>","<disgusted>","<furious>","<jealous>",
                "<impressed>","<surprised>"]
MODEL_INPUTS = ["input_ids","emo_intensity","knowledge_confidence", "position_ids","token_type_ids","vm","tag","attention_mask", "lm_labels_first","lm_labels_second","lm_labels_final",
                "decoder_input_ids_first","decoder_input_ids_second",  "decoder_input_ids_final",
                "decoder_attention_mask_first","decoder_attention_mask_second","decoder_attention_mask_final","emo_label"]  # , "token_type_ids","mc_token_ids", "mc_labels"
PADDED_INPUTS = ["input_ids", "decoder_input_ids_first", "lm_labels_first","lm_kg_first",
                 "decoder_input_ids_second", "lm_labels_second", "lm_kg_second",
                 "decoder_input_ids_final", "lm_labels_final","emo_token_ids"]
#inputids,pos,vm在 add_knowledge_with
SPECIAL_TOKENS=["<speaker1>","<speaker2>"]
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
logger = logging.getLogger(__file__)

NRC_CLASS={}
nrc=["<anger>","<anticipation>","<disgust>","<fear>","<joy>","<sadness>","<surprise>","<trust>","<other>"]
for i,k in enumerate(nrc):
    NRC_CLASS[k]=i
PAD_EMO_ID=NRC_CLASS["<other>"]

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
def get_losses_weights(losses:[list, np.ndarray, torch.Tensor]):
    if type(losses) != torch.Tensor:
        losses = torch.tensor(losses)
    weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
    return weights

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    '''From fairseq'''
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def average_distributed_scalar(scalar, config):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if config.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=config.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def train():
    config_file = "configs/train_pipline_config.json"
    config = Config.from_json_file(config_file)

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if config.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d",config.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(config))

    # Initialize distributed training if needed
    config.distributed = (config.local_rank != -1)
    if config.distributed:
        torch.cuda.set_device(config.local_rank)
        config.device = torch.device("cuda", config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    emo_labels_dict = {}
    for i, emo in enumerate(EMO_LABELS):
        emo_labels_dict[emo] = i
    # set_seed(config)
    tokenizer = BartTokenizer.from_pretrained(config.model_checkpoint)
    tokenizer.set_special_tokens(SPECIAL_TOKENS) # not use in stage two
    model = StepBartForDialogueGeneration  # (bart_config)
    model = model.from_pretrained(config.model_checkpoint)
    model.resize_token_embeddings(len(tokenizer)) #not use in stage two
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr) #model.integration.parameters in stage two
    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if config.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16)
    if config.distributed:
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(config, tokenizer,emo_labels_dict,MODEL_INPUTS)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(config.device) for input_tensor in batch)
        input_ids,emo_intensity,knowledge_confidence, position_ids,token_type_ids,vm,tag,attention_mask,lm_label_first, lm_label_second,lm_label_final,\
        decoder_input_ids_first,decoder_input_ids_second,decoder_input_ids_final, \
        decoder_attention_mask_first, decoder_attention_mask_second, decoder_attention_mask_final,\
        emo_label= batch
        output_first, output_second,output_final,emo_logits,topic_similarity = model(input_ids=input_ids,
                                                                    emo_intensity=emo_intensity,
                                                                    knowledge_confidence=knowledge_confidence,
                                                                    token_type_ids=token_type_ids,
                                                                    position_ids=position_ids,
                                                                    vm=vm,
                                                        attention_mask=tag,
                                                        decoder_input_ids_first=decoder_input_ids_first,
                                                        decoder_input_ids_second=decoder_input_ids_second,
                                                        decoder_input_ids_final=decoder_input_ids_final,
                                                        decoder_attention_mask_first=decoder_attention_mask_first,
                                                        decoder_attention_mask_second=decoder_attention_mask_second,
                                                        decoder_attention_mask_final=decoder_attention_mask_final,
                                                       is_train=True,
                                                       is_integrate=True,
                                                        hard_attention=False,
                                                       )
        lm_logits_first = output_first
        lm_logits_first_flat_shifted = lm_logits_first.contiguous().view(-1, lm_logits_first.size(-1))
        lm_labels_first_flat_shifted = lm_label_first.contiguous().view(-1)
        loss1 = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(lm_logits_first_flat_shifted, lm_labels_first_flat_shifted)
        lm_logits_second = output_second
        lm_logits_second_flat_shifted = lm_logits_second.contiguous().view(-1, lm_logits_second.size(-1))
        lm_labels_second_flat_shifted = lm_label_second.contiguous().view(-1)
        loss2 = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(lm_logits_second_flat_shifted, lm_labels_second_flat_shifted)
        lm_logits_final=output_final
        lm_logits_final_flat_shifted = lm_logits_final.contiguous().view(-1, lm_logits_final.size(-1))
        lm_labels_final_flat_shifted = lm_label_final.contiguous().view(-1)
        loss_final = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(lm_logits_final_flat_shifted,lm_labels_final_flat_shifted)
        emo_logits_flat_shifted = emo_logits.contiguous().view(-1, emo_logits.size(-1))
        emo_labels_flat_shifted = emo_label.contiguous().view(-1)
        loss_emo = torch.nn.CrossEntropyLoss()(emo_logits_flat_shifted,emo_labels_flat_shifted)
        loss = loss_emo +0.5* loss1 + 0.5*loss2# +0.5*loss_emo_words+0.5*loss_concept_words#loss_emo# / config.gradient_accumulation_steps
        # loss=loss_final # stage two
        if config.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        if engine.state.iteration % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(),loss_emo.item()

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(config.device) for input_tensor in batch)
            input_ids,emo_intensity,knowledge_confidence, position_ids,token_type_ids,vm,tag,attention_mask, lm_label_first, lm_label_second, lm_label_final, \
            decoder_input_ids_first, decoder_input_ids_second, decoder_input_ids_final, \
            decoder_attention_mask_first, decoder_attention_mask_second, decoder_attention_mask_final,\
            emo_label = batch
            output_first, output_second,output_final,emo_logits,topic_similarity = model(input_ids=input_ids,
                                                                            emo_intensity =None,#emo_intensity,
                                                                            knowledge_confidence=None,#knowledge_confidence,
                                                                                        token_type_ids=token_type_ids,
                                                                        position_ids=position_ids,
                                                                        vm=None,
                                                             attention_mask=tag,
                                                             decoder_input_ids_first=decoder_input_ids_first,
                                                             decoder_input_ids_second=decoder_input_ids_second,
                                                             decoder_input_ids_final=decoder_input_ids_final,
                                                             decoder_attention_mask_first=decoder_attention_mask_first,
                                                             decoder_attention_mask_second=decoder_attention_mask_second,
                                                             decoder_attention_mask_final=decoder_attention_mask_final,
                                                                        is_integrate=True,
                                                           is_train=True,
                                                            hard_attention=False
                                                           )
            lm_logits_first = output_first
            lm_logits_first_flat_shifted = lm_logits_first.contiguous().view(-1, lm_logits_first.size(-1))
            lm_labels_first_flat_shifted = lm_label_first.contiguous().view(-1)
            lm_logits_second = output_second
            lm_logits_second_flat_shifted = lm_logits_second.contiguous().view(-1, lm_logits_second.size(-1))
            lm_labels_second_flat_shifted = lm_label_second.contiguous().view(-1)
            lm_logits_final = output_final
            lm_logits_final_flat_shifted = lm_logits_final.contiguous().view(-1, lm_logits_final.size(-1))
            lm_labels_final_flat_shifted = lm_label_final.contiguous().view(-1)
            emo_logits_flat_shifted = emo_logits.contiguous().view(-1, emo_logits.size(-1))
            emo_labels_flat_shifted = emo_label.contiguous().view(-1)
            return (lm_logits_first_flat_shifted, lm_logits_second_flat_shifted,lm_logits_final_flat_shifted,emo_logits_flat_shifted), (
            lm_labels_first_flat_shifted, lm_labels_second_flat_shifted,lm_labels_final_flat_shifted,emo_labels_flat_shifted)

    trainer = Engine(update)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if config.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if config.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if config.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(4* len(train_loader), config.lr), (config.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "loss_emo")

    metrics = {"nll1": Loss(torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID), output_transform=lambda x: (x[0][0], x[1][0])),
               "nll2": Loss(torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID), output_transform=lambda x: (x[0][1], x[1][1])),
                "nll3": Loss(torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID), output_transform=lambda x: (x[0][2], x[1][2])),
               "nllemo": Loss(torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID),
                            output_transform=lambda x: (x[0][3], x[1][3])),
               # "precision":Precision(output_transform=lambda x: (x[0][4],x[1][4]),is_multilabel=True),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][3], x[1][3]))}
    metrics.update({"average_nll1": MetricsLambda(average_distributed_scalar, metrics["nll1"], config),
                    "average_nll2": MetricsLambda(average_distributed_scalar, metrics["nll2"], config),
                    "average_nll3": MetricsLambda(average_distributed_scalar, metrics["nll3"], config),
                    "average_nllemo": MetricsLambda(average_distributed_scalar, metrics["nllemo"], config),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], config)})
    metrics["average_ppl1"] = MetricsLambda(math.exp, metrics["average_nll1"])
    metrics["average_ppl2"] = MetricsLambda(math.exp, metrics["average_nll2"])
    metrics["average_ppl3"] = MetricsLambda(math.exp, metrics["average_nll3"])
    metrics["average_ppl"] = MetricsLambda(math.exp,(metrics["nll1"]+ metrics["nll2"]+metrics["nll3"])/3)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if config.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss","loss_emo"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=config.log_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                              another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(config.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(config, config.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(config.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(config.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=config.n_epochs)
    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if config.local_rank in [-1, 0] and config.n_epochs > 0:
        # print(checkpoint_handler._saved[-1][1])
        os.rename(os.path.join(config.log_dir, checkpoint_handler._saved[-1][1]), os.path.join(config.log_dir,
                                                                                               WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
