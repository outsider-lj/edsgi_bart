# coding=utf-8
import torch
import torch.nn as nn
from modeling import RobertaTokenizer,RobertaForSequenceClassification
from bart import StepBartForDialogueGeneration,StepFocusedModel
class MultiTask(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.model_attn_distribution=StepFocusedModel(config)#.from_pretrained(config.attn_distribution_model_path)
        self.model_step_generation=StepBartForDialogueGeneration(config)#.from_pretrained(config.step_generation_model_path)
        # self.tokenizer=RobertaTokenizer(config.tokenizer_path)
    def forward(self,
                input_ids=None,
                situation_ids=None,
                situation_mask=None,
                emo_intensity=None,
                knowledge_confidence=None,
            emo_input_ids=None,
            mc_token_ids=None,
        position_ids=None,
            vm=None,
        attention_mask=None,
        encoder_token_type_ids=None,
        decoder_input_ids_first=None,
        decoder_input_ids_second=None,
        decoder_input_ids_final=None,
            emo_input1_ids=None,
            emo_input2_ids=None,
            emo_inputfinal_ids=None,
        decoder_attention_mask_first=None,
        decoder_attention_mask_second=None,
        decoder_attention_mask_final=None,
        labels=None,
            is_train=True,
            is_integrate=False,
                hard_attention=False):
        logits,attentions = self.model_attn_distribution(input_ids=(input_ids, situation_ids),
                        attention_mask=(attention_mask, situation_mask),
                        vm=vm,
                        position_ids=(position_ids,None),
                        token_type_ids=(None, None),
                        return_dict=True)  # 进入model的forward
        emo_attention=attentions[1]
        topic_attention=attentions[0]
        emo_logits=logits[1]
        topic_logits=logits[0]
        if knowledge_confidence is not None and emo_intensity is not None:
            topic_hard_att=torch.eq(knowledge_confidence,1)
            emo_hard_att=emo_intensity
            if hard_attention is True:
                emo_attention=emo_hard_att
                topic_attention=topic_hard_att
        outputoutput_first, output_second,output_final = self.model_step_generation(input_ids=input_ids,
                                                                    # emo_input_ids=None,#emo_input_ids,
                                                                    # mc_token_ids=emo_token_ids,
                                                                    position_ids=position_ids,
                                                                    vm=vm,
                                                        attention_mask=attention_mask,
                                                        decoder_input_ids_first=decoder_input_ids_first,
                                                        decoder_input_ids_second=decoder_input_ids_second,
                                                        decoder_input_ids_final=decoder_input_ids_final,
                                                                    emo_input1_ids=None,#emo_input1_ids,
                                                                    emo_input2_ids=None,#emo_input2_ids,
                                                                    emo_inputfinal_ids=None,#emo_inputfinal_ids,
                                                        decoder_attention_mask_first=decoder_attention_mask_first,
                                                        decoder_attention_mask_second=decoder_attention_mask_second,
                                                        decoder_attention_mask_final=decoder_attention_mask_final,
                                                       emotion_focused_attention=emo_attention,
                                                    topic_focused_attention=topic_attention,
                                                       is_train=is_train,
                                                       is_integrate=is_integrate)
        return outputoutput_first, output_second,output_final,emo_logits,topic_logits