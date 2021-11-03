# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import sys
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig
from torch.nn import BCEWithLogitsLoss, CosineEmbeddingLoss,CrossEntropyLoss, MSELoss

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, text_d=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_d = text_d
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
    def _read_from_command(cls, claim, perspective, opp_perspective, label):
        lines = [claim, perspective, opp_perspective, label]
        return lines


import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_sm")

def generate_opp_dataset(df):

    matcher = PhraseMatcher(nlp.vocab, attr="lower")
    matcher.add("not", None, nlp("not"), nlp("n't"))
    matcher.add("can't", None, nlp("can't"))


    matcher_positive = PhraseMatcher(nlp.vocab, attr="ORTH")
    matcher_positive.add("type", None, nlp("is a type of"), nlp("are a type of") )
    matcher_positive.add("imply", None, nlp("implies") )
    matcher_positive.add("same", None, nlp("is the same as "), nlp("are the same as ") )
    matcher_positive.add("rephrase", None, nlp(" is a rephrasing of") )
    matcher_positive.add("form", None, nlp("is a another form of"))
    matcher_positive.add("synonym", None, nlp(" is synonymous with"))
    matcher_positive.add("can_be", None, nlp("can be"))


    matcher_positive.add("much", None, nlp("much"), nlp("Much"))
    matcher_positive.add("little", None, nlp("little"), nlp("Little"))
    matcher_positive.add("more", None, nlp("more"), nlp("More"))
    matcher_positive.add("less", None, nlp("less"), nlp("Less"))


    matcher_positive.add("fore_synonym", None, nlp("are synonyms"), nlp("are synonymous"), nlp("is the same thing"), nlp("are the same thing") )
    matcher_positive.add("then", None, nlp("then") )
    matcher_positive.add("so", None, nlp("so") )
    matcher_positive.add("must_be", None, nlp("must be"))
    matcher_positive.add("hasto", None, nlp("has to be"))
    matcher_positive.add("is_are", None, nlp("is"), nlp("are"), nlp("Are"), nlp("ARE"))

    matcher_positive.add("to", None, nlp("to"))
    matcher_positive.add("To", None, nlp("To"))
    matcher_positive.add("increase", None, nlp("increase"), nlp("increases"), nlp("Increase"))

    matcher_positive.add("should", None, nlp("should"), nlp("Should") )
    matcher_positive.add("would", None, nlp("would"), nlp("Would"))
    matcher_positive.add("could", None, nlp("could"), nlp("Could"))
    matcher_positive.add("may", None, nlp("may"), nlp("May"))
    matcher_positive.add("will", None, nlp("will"), nlp("Will"))
    matcher_positive.add("can", None, nlp("can"), nlp("Can"))
    matcher_positive.add("might", None, nlp("might"), nlp("Might"))
    matcher_positive.add("must", None, nlp("must"), nlp("Must"), nlp("MUST"))

    matcher_positive.add("encourage", None, nlp("encourage"), nlp("encourages"), nlp("Encourage"), nlp("Encourages"))

    matcher_positive.add("n’t", None, nlp("n’t"))
    matcher_positive.add("was_were", None, nlp("was"), nlp("were"))
    matcher_positive.add("raise", None, nlp("raise"), nlp("raises"), nlp("Raise"), nlp("raising"), nlp("Raising"))
    matcher_positive.add("better", None, nlp("better"), nlp("Better"))
    matcher_positive.add("benefit", None, nlp("benefit"), nlp("benefits"), nlp("Benefit"), nlp("Benefits"))
    matcher_positive.add("lack", None, nlp("lack"), nlp("lacks"), nlp("Lack"), nlp("Lacks"))
    matcher_positive.add("nothing", None, nlp("nothing"), nlp("Nothing"))
    matcher_positive.add("positive", None, nlp("positive"),nlp("Positive"))
    matcher_positive.add("negative", None, nlp("negative"), nlp("Negative"))
    matcher_positive.add("have", None, nlp("have"),nlp("Have"))
    matcher_positive.add("has", None, nlp("has"), nlp("Has"))
    matcher_positive.add("reduce", None, nlp("reduce"), nlp("Reduce"), nlp("reduces"), nlp("Reduces"), nlp("Reduced"), nlp("reduced"))
    matcher_positive.add("increase", None, nlp("increase"), nlp("Increase"), nlp("increases"), nlp("Increases"), nlp("increased"), nlp("Increased"))
    matcher_positive.add("without", None, nlp("without"), nlp("Without"))
    matcher_positive.add("against", None, nlp("against"), nlp("Against"))
    matcher_positive.add("need", None, nlp("need"), nlp("Need"), nlp("needs"), nlp("Needs"), nlp("needed"), nlp("Needed"))
    matcher_positive.add("needed", None, nlp("needed"), nlp("Needed"))
    matcher_positive.add("good", None, nlp("good"), nlp("Good"))
    matcher_positive.add("bad", None, nlp("bad"), nlp("Bad"))
    matcher_positive.add("support", None, nlp("support"), nlp("Support"), nlp("supports"), nlp("Supports"), nlp("supported"), nlp("Supported"))
    matcher_positive.add("hurt_harm_damage", None, nlp("hurt"), nlp("hurts"), nlp("harm"), nlp("harms"), nlp("Hurt"), nlp("Hurts"), nlp("Harm"), nlp("Harms"), nlp("HARM"), nlp("damage"), nlp("damages"), nlp("Damage"), nlp("Damages"))
    matcher_positive.add("help", None, nlp("help"), nlp("Help"), nlp("helps"), nlp("Helps"))
    matcher_positive.add("protect", None, nlp("protect"), nlp("Protect"), nlp("protects"), nlp("Protects"), nlp("protecting"), nlp("protected"), nlp("Protecting"), nlp("Protected"))
    matcher_positive.add("cause", None, nlp("cause"), nlp("Cause"), nlp("causes"), nlp("Causes"))
    matcher_positive.add("allow", None, nlp("allow"), nlp("Allow"), nlp("allows"), nlp("Allows"))
    matcher_positive.add("everyone", None, nlp("everyone"), nlp("Everyone"))
    matcher_positive.add("deserve", None, nlp("deserve"), nlp("Deserve"), nlp("deserves"), nlp("Deserves"), nlp("deserved"), nlp("Deserved"))

    opposite = []
    for claim in df.claim:
        doc = nlp(claim)
        matches = matcher(doc)
        positive_matches = matcher_positive(doc)
        if matches:
            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "can't":
                    new_seq = str(doc[0:start-1])+" can "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "not":
                    if str(doc[start-1:start]) == "ca":
                        continue
                    else:
                        new_seq = str(doc[0:start])+" "+str(doc[end:])
                        opposite.append(new_seq)
                        break
        elif positive_matches:
            for match_id, start, end in positive_matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "type":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "imply":
                    if str(doc[start:end][0]) == "implies":
                        new_seq = str(doc[0:start])+" does not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif str(doc[start:end][0]) == "imply":
                        new_seq = str(doc[0:start])+" do not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "same":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "rephrase":
                    new_seq = str(doc[0:start])+" is not a rephrasing of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "form":
                    new_seq = str(doc[0:start])+" is a another form of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "synonym":
                    new_seq = str(doc[0:start])+" is not synonymous with "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "can_be":
                    new_seq = str(doc[0:start])+" can't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "fore_synonym":
                    new_seq = str(doc[0:start])+" are not synonymous"
                    opposite.append(new_seq)
                    break 
                elif rule_id == "then":
                    new_seq = str(doc[0:start])+" doesn't mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "so":
                    new_seq = str(doc[0:start])+" does not mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "must_be":
                    new_seq = str(doc[0:start])+" needn't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "hasto":
                    new_seq = str(doc[0:start])+" doesn't have to be"+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "much":
                    if str(doc[start:end][0]) == "Much":
                        new_seq = str("Little "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "much":
                        new_seq = str(doc[0:start])+" little "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "little":
                    if str(doc[start:end][0]) == "Little":
                        new_seq = str("Much "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "little":
                        new_seq = str(doc[0:start])+" much "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "more":
                    if str(doc[start:end][0]) == "More":
                        new_seq = str("Less "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "more":
                        new_seq = str(doc[0:start])+" less "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "less":
                    if str(doc[start:end][0]) == "Less":
                        new_seq = str("More "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "less":
                        new_seq = str(doc[0:start])+" more "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "is_are":
                    if str(doc[start:end][0]) == "is":
                        new_seq = str(doc[0:start])+" is not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "Are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "ARE":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "to":
                    new_seq = str(doc[0:start])+" not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "To":
                    new_seq = str(doc[0:start])+" Not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "should":
                    new_seq = str(doc[0:start])+" should not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "would":
                    new_seq = str(doc[0:start])+" would not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "could":
                    new_seq = str(doc[0:start])+" could not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "may":
                    new_seq = str(doc[0:start])+" may not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "will":
                    new_seq = str(doc[0:start])+" will not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "can":
                    new_seq = str(doc[0:start])+" can not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "might":
                    new_seq = str(doc[0:start])+" might not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "must":
                    new_seq = str(doc[0:start])+" must not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "encourage":
                    new_seq = str(doc[0:start])+" discourage "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "n’t":
                    new_seq = str(doc[0:start])+" "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "was_were":
                    if str(doc[start:end][0]) == "was":
                        new_seq = str(doc[0:start])+" was not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "were":
                        new_seq = str(doc[0:start])+" were not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "raise":
                    new_seq = str(doc[0:start])+" lower "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "better":
                    new_seq = str(doc[0:start])+" worse "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "benefit":
                    new_seq = str(doc[0:start])+" harm "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "lack":
                    new_seq = str(doc[0:start])+" glut "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "nothing":
                    new_seq = str(doc[0:start])+" something "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "positive":
                    new_seq = str(doc[0:start])+" negative "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "negative":
                    new_seq = str(doc[0:start])+" positive "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "have":
                    new_seq = str(doc[0:start])+" don't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "has":
                    new_seq = str(doc[0:start])+" doesn't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "reduce":
                    new_seq = str(doc[0:start])+" increase "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "without":
                    new_seq = str(doc[0:start])+" with "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "against":
                    new_seq = str(doc[0:start])+" for "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "needed":
                    new_seq = str(doc[0:start])+" not needed "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "need":
                    new_seq = str(doc[0:start])+" don't need "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "good":
                    new_seq = str(doc[0:start])+" bad "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "bad":
                    new_seq = str(doc[0:start])+" good "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "support":
                    new_seq = str(doc[0:start])+" oppose "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "hurt_harm_damage":
                    new_seq = str(doc[0:start])+" protect "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "help":
                    new_seq = str(doc[0:start])+" spoil "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "protect":
                    new_seq = str(doc[0:start])+" destroy "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "cause":
                    new_seq = str(doc[0:start])+" casue no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "allow":
                    new_seq = str(doc[0:start])+" disallow "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "everyone":
                    new_seq = str(doc[0:start])+" noone "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "deserve":
                    new_seq = str(doc[0:start])+" deserve no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
        else:
            new_seq = None
            opposite.append(new_seq)
            
    df["opposite_claim"] = opposite
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    opposite = []
    for perspective in df.perspective:
        doc = nlp(perspective)
        matches = matcher(doc)
        positive_matches = matcher_positive(doc)
        if matches:
            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "can't":
                    new_seq = str(doc[0:start-1])+" can "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "not":
                    if str(doc[start-1:start]) == "ca":
                        continue
                    else:
                        new_seq = str(doc[0:start])+" "+str(doc[end:])
                        opposite.append(new_seq)
                        break
        elif positive_matches:
            for match_id, start, end in positive_matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "type":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "imply":
                    if str(doc[start:end][0]) == "implies":
                        new_seq = str(doc[0:start])+" does not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif str(doc[start:end][0]) == "imply":
                        new_seq = str(doc[0:start])+" do not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "same":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "rephrase":
                    new_seq = str(doc[0:start])+" is not a rephrasing of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "form":
                    new_seq = str(doc[0:start])+" is a another form of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "synonym":
                    new_seq = str(doc[0:start])+" is not synonymous with "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "can_be":
                    new_seq = str(doc[0:start])+" can't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "fore_synonym":
                    new_seq = str(doc[0:start])+" are not synonymous"
                    opposite.append(new_seq)
                    break 
                elif rule_id == "then":
                    new_seq = str(doc[0:start])+" doesn't mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "so":
                    new_seq = str(doc[0:start])+" does not mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "must_be":
                    new_seq = str(doc[0:start])+" needn't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "hasto":
                    new_seq = str(doc[0:start])+" doesn't have to be"+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "much":
                    if str(doc[start:end][0]) == "Much":
                        new_seq = str("Little "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "much":
                        new_seq = str(doc[0:start])+" little "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "little":
                    if str(doc[start:end][0]) == "Little":
                        new_seq = str("Much "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "little":
                        new_seq = str(doc[0:start])+" much "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "more":
                    if str(doc[start:end][0]) == "More":
                        new_seq = str("Less "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "more":
                        new_seq = str(doc[0:start])+" less "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "less":
                    if str(doc[start:end][0]) == "Less":
                        new_seq = str("More "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "less":
                        new_seq = str(doc[0:start])+" more "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "is_are":
                    if str(doc[start:end][0]) == "is":
                        new_seq = str(doc[0:start])+" is not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "Are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "ARE":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "to":
                    new_seq = str(doc[0:start])+" not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "To":
                    new_seq = str(doc[0:start])+" Not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "should":
                    new_seq = str(doc[0:start])+" should not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "would":
                    new_seq = str(doc[0:start])+" would not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "could":
                    new_seq = str(doc[0:start])+" could not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "may":
                    new_seq = str(doc[0:start])+" may not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "will":
                    new_seq = str(doc[0:start])+" will not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "can":
                    new_seq = str(doc[0:start])+" can not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "might":
                    new_seq = str(doc[0:start])+" might not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "must":
                    new_seq = str(doc[0:start])+" must not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "encourage":
                    new_seq = str(doc[0:start])+" discourage "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "n’t":
                    new_seq = str(doc[0:start])+" "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "was_were":
                    if str(doc[start:end][0]) == "was":
                        new_seq = str(doc[0:start])+" was not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "were":
                        new_seq = str(doc[0:start])+" were not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "raise":
                    new_seq = str(doc[0:start])+" lower "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "better":
                    new_seq = str(doc[0:start])+" worse "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "benefit":
                    new_seq = str(doc[0:start])+" harm "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "lack":
                    new_seq = str(doc[0:start])+" glut "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "nothing":
                    new_seq = str(doc[0:start])+" something "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "positive":
                    new_seq = str(doc[0:start])+" negative "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "negative":
                    new_seq = str(doc[0:start])+" positive "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "have":
                    new_seq = str(doc[0:start])+" don't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "has":
                    new_seq = str(doc[0:start])+" doesn't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "reduce":
                    new_seq = str(doc[0:start])+" increase "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "without":
                    new_seq = str(doc[0:start])+" with "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "against":
                    new_seq = str(doc[0:start])+" for "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "needed":
                    new_seq = str(doc[0:start])+" not needed "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "need":
                    new_seq = str(doc[0:start])+" don't need "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "good":
                    new_seq = str(doc[0:start])+" bad "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "bad":
                    new_seq = str(doc[0:start])+" good "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "support":
                    new_seq = str(doc[0:start])+" oppose "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "hurt_harm_damage":
                    new_seq = str(doc[0:start])+" protect "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "help":
                    new_seq = str(doc[0:start])+" spoil "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "protect":
                    new_seq = str(doc[0:start])+" destroy "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "cause":
                    new_seq = str(doc[0:start])+" casue no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "allow":
                    new_seq = str(doc[0:start])+" disallow "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "everyone":
                    new_seq = str(doc[0:start])+" noone "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "deserve":
                    new_seq = str(doc[0:start])+" deserve no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
        else:
            new_seq = None
            opposite.append(new_seq)
    df["opposite_perspective"] = opposite
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def generate_opp_pers_dataset(df):

    matcher = PhraseMatcher(nlp.vocab, attr="lower")
    matcher.add("not", None, nlp("not"), nlp("n't"))
    matcher.add("can't", None, nlp("can't"))


    matcher_positive = PhraseMatcher(nlp.vocab, attr="ORTH")
    matcher_positive.add("type", None, nlp("is a type of"), nlp("are a type of") )
    matcher_positive.add("imply", None, nlp("implies") )
    matcher_positive.add("same", None, nlp("is the same as "), nlp("are the same as ") )
    matcher_positive.add("rephrase", None, nlp(" is a rephrasing of") )
    matcher_positive.add("form", None, nlp("is a another form of"))
    matcher_positive.add("synonym", None, nlp(" is synonymous with"))
    matcher_positive.add("can_be", None, nlp("can be"))


    matcher_positive.add("much", None, nlp("much"), nlp("Much"))
    matcher_positive.add("little", None, nlp("little"), nlp("Little"))
    matcher_positive.add("more", None, nlp("more"), nlp("More"))
    matcher_positive.add("less", None, nlp("less"), nlp("Less"))


    matcher_positive.add("fore_synonym", None, nlp("are synonyms"), nlp("are synonymous"), nlp("is the same thing"), nlp("are the same thing") )
    matcher_positive.add("then", None, nlp("then") )
    matcher_positive.add("so", None, nlp("so") )
    matcher_positive.add("must_be", None, nlp("must be"))
    matcher_positive.add("hasto", None, nlp("has to be"))
    matcher_positive.add("is_are", None, nlp("is"), nlp("are"), nlp("Are"), nlp("ARE"))

    matcher_positive.add("to", None, nlp("to"))
    matcher_positive.add("To", None, nlp("To"))
    matcher_positive.add("increase", None, nlp("increase"), nlp("increases"), nlp("Increase"))

    matcher_positive.add("should", None, nlp("should"), nlp("Should") )
    matcher_positive.add("would", None, nlp("would"), nlp("Would"))
    matcher_positive.add("could", None, nlp("could"), nlp("Could"))
    matcher_positive.add("may", None, nlp("may"), nlp("May"))
    matcher_positive.add("will", None, nlp("will"), nlp("Will"))
    matcher_positive.add("can", None, nlp("can"), nlp("Can"))
    matcher_positive.add("might", None, nlp("might"), nlp("Might"))
    matcher_positive.add("must", None, nlp("must"), nlp("Must"), nlp("MUST"))

    matcher_positive.add("encourage", None, nlp("encourage"), nlp("encourages"), nlp("Encourage"), nlp("Encourages"))

    matcher_positive.add("n’t", None, nlp("n’t"))
    matcher_positive.add("was_were", None, nlp("was"), nlp("were"))
    matcher_positive.add("raise", None, nlp("raise"), nlp("raises"), nlp("Raise"), nlp("raising"), nlp("Raising"))
    matcher_positive.add("better", None, nlp("better"), nlp("Better"))
    matcher_positive.add("benefit", None, nlp("benefit"), nlp("benefits"), nlp("Benefit"), nlp("Benefits"))
    matcher_positive.add("lack", None, nlp("lack"), nlp("lacks"), nlp("Lack"), nlp("Lacks"))
    matcher_positive.add("nothing", None, nlp("nothing"), nlp("Nothing"))
    matcher_positive.add("positive", None, nlp("positive"),nlp("Positive"))
    matcher_positive.add("negative", None, nlp("negative"), nlp("Negative"))
    matcher_positive.add("have", None, nlp("have"),nlp("Have"))
    matcher_positive.add("has", None, nlp("has"), nlp("Has"))
    matcher_positive.add("reduce", None, nlp("reduce"), nlp("Reduce"), nlp("reduces"), nlp("Reduces"), nlp("Reduced"), nlp("reduced"))
    matcher_positive.add("increase", None, nlp("increase"), nlp("Increase"), nlp("increases"), nlp("Increases"), nlp("increased"), nlp("Increased"))
    matcher_positive.add("without", None, nlp("without"), nlp("Without"))
    matcher_positive.add("against", None, nlp("against"), nlp("Against"))
    matcher_positive.add("need", None, nlp("need"), nlp("Need"), nlp("needs"), nlp("Needs"), nlp("needed"), nlp("Needed"))
    matcher_positive.add("needed", None, nlp("needed"), nlp("Needed"))
    matcher_positive.add("good", None, nlp("good"), nlp("Good"))
    matcher_positive.add("bad", None, nlp("bad"), nlp("Bad"))
    matcher_positive.add("support", None, nlp("support"), nlp("Support"), nlp("supports"), nlp("Supports"), nlp("supported"), nlp("Supported"))
    matcher_positive.add("hurt_harm_damage", None, nlp("hurt"), nlp("hurts"), nlp("harm"), nlp("harms"), nlp("Hurt"), nlp("Hurts"), nlp("Harm"), nlp("Harms"), nlp("HARM"), nlp("damage"), nlp("damages"), nlp("Damage"), nlp("Damages"))
    matcher_positive.add("help", None, nlp("help"), nlp("Help"), nlp("helps"), nlp("Helps"))
    matcher_positive.add("protect", None, nlp("protect"), nlp("Protect"), nlp("protects"), nlp("Protects"), nlp("protecting"), nlp("protected"), nlp("Protecting"), nlp("Protected"))
    matcher_positive.add("cause", None, nlp("cause"), nlp("Cause"), nlp("causes"), nlp("Causes"))
    matcher_positive.add("allow", None, nlp("allow"), nlp("Allow"), nlp("allows"), nlp("Allows"))
    matcher_positive.add("everyone", None, nlp("everyone"), nlp("Everyone"))
    matcher_positive.add("deserve", None, nlp("deserve"), nlp("Deserve"), nlp("deserves"), nlp("Deserves"), nlp("deserved"), nlp("Deserved"))
    
    opposite = []
    for perspective in df.perspective:
        doc = nlp(perspective)
        matches = matcher(doc)
        positive_matches = matcher_positive(doc)
        if matches:
            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "can't":
                    new_seq = str(doc[0:start-1])+" can "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "not":
                    if str(doc[start-1:start]) == "ca":
                        continue
                    else:
                        new_seq = str(doc[0:start])+" "+str(doc[end:])
                        opposite.append(new_seq)
                        break
        elif positive_matches:
            for match_id, start, end in positive_matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "type":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "imply":
                    if str(doc[start:end][0]) == "implies":
                        new_seq = str(doc[0:start])+" does not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif str(doc[start:end][0]) == "imply":
                        new_seq = str(doc[0:start])+" do not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "same":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "rephrase":
                    new_seq = str(doc[0:start])+" is not a rephrasing of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "form":
                    new_seq = str(doc[0:start])+" is a another form of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "synonym":
                    new_seq = str(doc[0:start])+" is not synonymous with "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "can_be":
                    new_seq = str(doc[0:start])+" can't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "fore_synonym":
                    new_seq = str(doc[0:start])+" are not synonymous"
                    opposite.append(new_seq)
                    break 
                elif rule_id == "then":
                    new_seq = str(doc[0:start])+" doesn't mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "so":
                    new_seq = str(doc[0:start])+" does not mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "must_be":
                    new_seq = str(doc[0:start])+" needn't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "hasto":
                    new_seq = str(doc[0:start])+" doesn't have to be"+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "much":
                    if str(doc[start:end][0]) == "Much":
                        new_seq = str("Little "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "much":
                        new_seq = str(doc[0:start])+" little "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "little":
                    if str(doc[start:end][0]) == "Little":
                        new_seq = str("Much "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "little":
                        new_seq = str(doc[0:start])+" much "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "more":
                    if str(doc[start:end][0]) == "More":
                        new_seq = str("Less "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "more":
                        new_seq = str(doc[0:start])+" less "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "less":
                    if str(doc[start:end][0]) == "Less":
                        new_seq = str("More "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "less":
                        new_seq = str(doc[0:start])+" more "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "is_are":
                    if str(doc[start:end][0]) == "is":
                        new_seq = str(doc[0:start])+" is not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "Are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "ARE":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "to":
                    new_seq = str(doc[0:start])+" not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "To":
                    new_seq = str(doc[0:start])+" Not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "should":
                    new_seq = str(doc[0:start])+" should not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "would":
                    new_seq = str(doc[0:start])+" would not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "could":
                    new_seq = str(doc[0:start])+" could not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "may":
                    new_seq = str(doc[0:start])+" may not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "will":
                    new_seq = str(doc[0:start])+" will not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "can":
                    new_seq = str(doc[0:start])+" can not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "might":
                    new_seq = str(doc[0:start])+" might not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "must":
                    new_seq = str(doc[0:start])+" must not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "encourage":
                    new_seq = str(doc[0:start])+" discourage "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "n’t":
                    new_seq = str(doc[0:start])+" "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "was_were":
                    if str(doc[start:end][0]) == "was":
                        new_seq = str(doc[0:start])+" was not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "were":
                        new_seq = str(doc[0:start])+" were not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "raise":
                    new_seq = str(doc[0:start])+" lower "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "better":
                    new_seq = str(doc[0:start])+" worse "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "benefit":
                    new_seq = str(doc[0:start])+" harm "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "lack":
                    new_seq = str(doc[0:start])+" glut "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "nothing":
                    new_seq = str(doc[0:start])+" something "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "positive":
                    new_seq = str(doc[0:start])+" negative "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "negative":
                    new_seq = str(doc[0:start])+" positive "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "have":
                    new_seq = str(doc[0:start])+" don't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "has":
                    new_seq = str(doc[0:start])+" doesn't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "reduce":
                    new_seq = str(doc[0:start])+" increase "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "without":
                    new_seq = str(doc[0:start])+" with "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "against":
                    new_seq = str(doc[0:start])+" for "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "needed":
                    new_seq = str(doc[0:start])+" not needed "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "need":
                    new_seq = str(doc[0:start])+" don't need "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "good":
                    new_seq = str(doc[0:start])+" bad "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "bad":
                    new_seq = str(doc[0:start])+" good "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "support":
                    new_seq = str(doc[0:start])+" oppose "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "hurt_harm_damage":
                    new_seq = str(doc[0:start])+" protect "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "help":
                    new_seq = str(doc[0:start])+" spoil "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "protect":
                    new_seq = str(doc[0:start])+" destroy "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "cause":
                    new_seq = str(doc[0:start])+" casue no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "allow":
                    new_seq = str(doc[0:start])+" disallow "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "everyone":
                    new_seq = str(doc[0:start])+" noone "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "deserve":
                    new_seq = str(doc[0:start])+" deserve no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
        else:
            new_seq = None
            opposite.append(new_seq)
    df["opposite_perspective"] = opposite
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df



def generate_opp_pers_dataset_with_naive(df):

    matcher = PhraseMatcher(nlp.vocab, attr="lower")
    matcher.add("not", None, nlp("not"), nlp("n't"))
    matcher.add("can't", None, nlp("can't"))


    matcher_positive = PhraseMatcher(nlp.vocab, attr="ORTH")
    matcher_positive.add("type", None, nlp("is a type of"), nlp("are a type of") )
    matcher_positive.add("imply", None, nlp("implies") )
    matcher_positive.add("same", None, nlp("is the same as "), nlp("are the same as ") )
    matcher_positive.add("rephrase", None, nlp(" is a rephrasing of") )
    matcher_positive.add("form", None, nlp("is a another form of"))
    matcher_positive.add("synonym", None, nlp(" is synonymous with"))
    matcher_positive.add("can_be", None, nlp("can be"))


    matcher_positive.add("much", None, nlp("much"), nlp("Much"))
    matcher_positive.add("little", None, nlp("little"), nlp("Little"))
    matcher_positive.add("more", None, nlp("more"), nlp("More"))
    matcher_positive.add("less", None, nlp("less"), nlp("Less"))


    matcher_positive.add("fore_synonym", None, nlp("are synonyms"), nlp("are synonymous"), nlp("is the same thing"), nlp("are the same thing") )
    matcher_positive.add("then", None, nlp("then") )
    matcher_positive.add("so", None, nlp("so") )
    matcher_positive.add("must_be", None, nlp("must be"))
    matcher_positive.add("hasto", None, nlp("has to be"))
    matcher_positive.add("is_are", None, nlp("is"), nlp("are"), nlp("Are"), nlp("ARE"))

    matcher_positive.add("to", None, nlp("to"))
    matcher_positive.add("To", None, nlp("To"))
    matcher_positive.add("increase", None, nlp("increase"), nlp("increases"), nlp("Increase"))

    matcher_positive.add("should", None, nlp("should"), nlp("Should") )
    matcher_positive.add("would", None, nlp("would"), nlp("Would"))
    matcher_positive.add("could", None, nlp("could"), nlp("Could"))
    matcher_positive.add("may", None, nlp("may"), nlp("May"))
    matcher_positive.add("will", None, nlp("will"), nlp("Will"))
    matcher_positive.add("can", None, nlp("can"), nlp("Can"))
    matcher_positive.add("might", None, nlp("might"), nlp("Might"))
    matcher_positive.add("must", None, nlp("must"), nlp("Must"), nlp("MUST"))

    matcher_positive.add("encourage", None, nlp("encourage"), nlp("encourages"), nlp("Encourage"), nlp("Encourages"))

    matcher_positive.add("n’t", None, nlp("n’t"))
    matcher_positive.add("was_were", None, nlp("was"), nlp("were"))
    matcher_positive.add("raise", None, nlp("raise"), nlp("raises"), nlp("Raise"), nlp("raising"), nlp("Raising"))
    matcher_positive.add("better", None, nlp("better"), nlp("Better"))
    matcher_positive.add("benefit", None, nlp("benefit"), nlp("benefits"), nlp("Benefit"), nlp("Benefits"))
    matcher_positive.add("lack", None, nlp("lack"), nlp("lacks"), nlp("Lack"), nlp("Lacks"))
    matcher_positive.add("nothing", None, nlp("nothing"), nlp("Nothing"))
    matcher_positive.add("positive", None, nlp("positive"),nlp("Positive"))
    matcher_positive.add("negative", None, nlp("negative"), nlp("Negative"))
    matcher_positive.add("have", None, nlp("have"),nlp("Have"))
    matcher_positive.add("has", None, nlp("has"), nlp("Has"))
    matcher_positive.add("reduce", None, nlp("reduce"), nlp("Reduce"), nlp("reduces"), nlp("Reduces"), nlp("Reduced"), nlp("reduced"))
    matcher_positive.add("increase", None, nlp("increase"), nlp("Increase"), nlp("increases"), nlp("Increases"), nlp("increased"), nlp("Increased"))
    matcher_positive.add("without", None, nlp("without"), nlp("Without"))
    matcher_positive.add("against", None, nlp("against"), nlp("Against"))
    matcher_positive.add("need", None, nlp("need"), nlp("Need"), nlp("needs"), nlp("Needs"), nlp("needed"), nlp("Needed"))
    matcher_positive.add("needed", None, nlp("needed"), nlp("Needed"))
    matcher_positive.add("good", None, nlp("good"), nlp("Good"))
    matcher_positive.add("bad", None, nlp("bad"), nlp("Bad"))
    matcher_positive.add("support", None, nlp("support"), nlp("Support"), nlp("supports"), nlp("Supports"), nlp("supported"), nlp("Supported"))
    matcher_positive.add("hurt_harm_damage", None, nlp("hurt"), nlp("hurts"), nlp("harm"), nlp("harms"), nlp("Hurt"), nlp("Hurts"), nlp("Harm"), nlp("Harms"), nlp("HARM"), nlp("damage"), nlp("damages"), nlp("Damage"), nlp("Damages"))
    matcher_positive.add("help", None, nlp("help"), nlp("Help"), nlp("helps"), nlp("Helps"))
    matcher_positive.add("protect", None, nlp("protect"), nlp("Protect"), nlp("protects"), nlp("Protects"), nlp("protecting"), nlp("protected"), nlp("Protecting"), nlp("Protected"))
    matcher_positive.add("cause", None, nlp("cause"), nlp("Cause"), nlp("causes"), nlp("Causes"))
    matcher_positive.add("allow", None, nlp("allow"), nlp("Allow"), nlp("allows"), nlp("Allows"))
    matcher_positive.add("everyone", None, nlp("everyone"), nlp("Everyone"))
    matcher_positive.add("deserve", None, nlp("deserve"), nlp("Deserve"), nlp("deserves"), nlp("Deserves"), nlp("deserved"), nlp("Deserved"))
    
    opposite = []
    for perspective in df.perspective:
        doc = nlp(perspective)
        matches = matcher(doc)
        positive_matches = matcher_positive(doc)
        if matches:
            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "can't":
                    new_seq = str(doc[0:start-1])+" can "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "not":
                    if str(doc[start-1:start]) == "ca":
                        continue
                    else:
                        new_seq = str(doc[0:start])+" "+str(doc[end:])
                        opposite.append(new_seq)
                        break
        elif positive_matches:
            for match_id, start, end in positive_matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "type":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not a type of "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "imply":
                    if str(doc[start:end][0]) == "implies":
                        new_seq = str(doc[0:start])+" does not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    elif str(doc[start:end][0]) == "imply":
                        new_seq = str(doc[0:start])+" do not imply "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "same":
                    if doc[start:end][0] == "is":
                        new_seq = str(doc[0:start])+" is not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if doc[start:end][0] == "are":
                        new_seq = str(doc[0:start])+" are not the same as "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "rephrase":
                    new_seq = str(doc[0:start])+" is not a rephrasing of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "form":
                    new_seq = str(doc[0:start])+" is a another form of "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "synonym":
                    new_seq = str(doc[0:start])+" is not synonymous with "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "can_be":
                    new_seq = str(doc[0:start])+" can't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "fore_synonym":
                    new_seq = str(doc[0:start])+" are not synonymous"
                    opposite.append(new_seq)
                    break 
                elif rule_id == "then":
                    new_seq = str(doc[0:start])+" doesn't mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "so":
                    new_seq = str(doc[0:start])+" does not mean "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "must_be":
                    new_seq = str(doc[0:start])+" needn't be "+str(doc[end:])
                    opposite.append(new_seq)
                    break 
                elif rule_id == "hasto":
                    new_seq = str(doc[0:start])+" doesn't have to be"+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "much":
                    if str(doc[start:end][0]) == "Much":
                        new_seq = str("Little "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "much":
                        new_seq = str(doc[0:start])+" little "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "little":
                    if str(doc[start:end][0]) == "Little":
                        new_seq = str("Much "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "little":
                        new_seq = str(doc[0:start])+" much "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "more":
                    if str(doc[start:end][0]) == "More":
                        new_seq = str("Less "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "more":
                        new_seq = str(doc[0:start])+" less "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "less":
                    if str(doc[start:end][0]) == "Less":
                        new_seq = str("More "+str(doc[end:]))
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "less":
                        new_seq = str(doc[0:start])+" more "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "is_are":
                    if str(doc[start:end][0]) == "is":
                        new_seq = str(doc[0:start])+" is not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "Are":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "ARE":
                        new_seq = str(doc[0:start])+" are not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "to":
                    new_seq = str(doc[0:start])+" not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "To":
                    new_seq = str(doc[0:start])+" Not to "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "should":
                    new_seq = str(doc[0:start])+" should not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "would":
                    new_seq = str(doc[0:start])+" would not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "could":
                    new_seq = str(doc[0:start])+" could not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "may":
                    new_seq = str(doc[0:start])+" may not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "will":
                    new_seq = str(doc[0:start])+" will not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "can":
                    new_seq = str(doc[0:start])+" can not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "might":
                    new_seq = str(doc[0:start])+" might not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "must":
                    new_seq = str(doc[0:start])+" must not "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "encourage":
                    new_seq = str(doc[0:start])+" discourage "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "n’t":
                    new_seq = str(doc[0:start])+" "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "was_were":
                    if str(doc[start:end][0]) == "was":
                        new_seq = str(doc[0:start])+" was not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                    if str(doc[start:end][0]) == "were":
                        new_seq = str(doc[0:start])+" were not "+str(doc[end:])
                        opposite.append(new_seq)
                        break
                elif rule_id == "raise":
                    new_seq = str(doc[0:start])+" lower "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "better":
                    new_seq = str(doc[0:start])+" worse "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "benefit":
                    new_seq = str(doc[0:start])+" harm "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "lack":
                    new_seq = str(doc[0:start])+" glut "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "nothing":
                    new_seq = str(doc[0:start])+" something "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "positive":
                    new_seq = str(doc[0:start])+" negative "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "negative":
                    new_seq = str(doc[0:start])+" positive "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "have":
                    new_seq = str(doc[0:start])+" don't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "has":
                    new_seq = str(doc[0:start])+" doesn't have "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "reduce":
                    new_seq = str(doc[0:start])+" increase "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "increase":
                    new_seq = str(doc[0:start])+" decrease "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "without":
                    new_seq = str(doc[0:start])+" with "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "against":
                    new_seq = str(doc[0:start])+" for "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "needed":
                    new_seq = str(doc[0:start])+" not needed "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "need":
                    new_seq = str(doc[0:start])+" don't need "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "good":
                    new_seq = str(doc[0:start])+" bad "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "bad":
                    new_seq = str(doc[0:start])+" good "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "support":
                    new_seq = str(doc[0:start])+" oppose "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "hurt_harm_damage":
                    new_seq = str(doc[0:start])+" protect "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "help":
                    new_seq = str(doc[0:start])+" spoil "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "protect":
                    new_seq = str(doc[0:start])+" destroy "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "cause":
                    new_seq = str(doc[0:start])+" casue no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "allow":
                    new_seq = str(doc[0:start])+" disallow "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "everyone":
                    new_seq = str(doc[0:start])+" noone "+str(doc[end:])
                    opposite.append(new_seq)
                    break
                elif rule_id == "deserve":
                    new_seq = str(doc[0:start])+" deserve no "+str(doc[end:])
                    opposite.append(new_seq)
                    break
        else:
#             new_seq = None
            new_seq = str(doc)+" but it is not true. "
            opposite.append(new_seq)
    df["opposite_perspective"] = opposite
#     df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
    
    
import pandas as pd   
class NegProcessor(DataProcessor):
    """Processor for the Perspectrum data set ."""
    
    def get_train_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")

    def get_test_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t")

    def get_dev_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep="\t")
    
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "new_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "new_train.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "new_test.tsv")), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "new_dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            text_c = line[5]
            text_d = line[6]
            label = line[0]
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples

    
    
class TriProcessor(DataProcessor):
    """Processor for the Perspectrum data set ."""
    
    def get_train_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")

    def get_test_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t")

    def get_dev_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep="\t")
    
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "new_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "tri_train.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "tri_test.tsv")), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "tri_dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    
    def get_test_from_command(self, claim, perspective, opp_perspective, label):
        """get sequence from command"""
        return self._create_examples_from_command(
            self._read_from_command(claim, perspective, opp_perspective, label), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            text_c = line[5]
            text_d = None
            label = line[0]
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples
    
    def _create_examples_from_command(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
       
        guid = "%s-%s" % (set_type, 1)
        text_a = lines[0]
        text_b = lines[1]
        text_c = lines[2]
        text_d = None
        label = lines[3]

        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples
    
    

class StanceProcessor(DataProcessor):
    """Processor for the Perspectrum data set ."""
    
    def get_train_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")

    def get_test_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t")

    def get_dev_df(self, data_dir):
        """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep="\t")
    
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            text_c = None
            label = line[0]
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
        return examples
    
        
        
        
        
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            text_c = None
            label = line[0]
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


"""""
Customized fucntion start

"""""


def convert_claims_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Claim Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def convert_opp_claims_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_c)

        tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Opposite Claim Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def convert_opposite_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

#         tokens_b = None
        
        tokens_c = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)
        if example.text_c:
            tokens_c = tokenizer.tokenize(example.text_c)
#         if tokens_b:
#             # Modifies `tokens_a` and `tokens_b` in place so that the total
#             # length is less than the specified length.
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        if tokens_c:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_c, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_c:
            for token in tokens_c:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Opposite Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features



def convert_pers_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_b)

        tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Perspective Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def convert_opp_pers_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_d)

        tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Opposite Perspective Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features



def convert_triopp_pers_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_c)

        tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Opposite Perspective Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


"""""
Customized fucntion end

"""""

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def p_r_f1(out, labels):
    outputs = np.argmax(out, axis=1)
    tp = np.sum(np.dot(outputs, labels))
    pred_count = np.sum(outputs == 1)
    gold_count = np.sum(labels == 1)

    if pred_count == 0:
        p = 1
    else:
        p = tp / pred_count

    if gold_count == 0:
        r = 1
    else:
        r = tp / gold_count

    # print(outputs, labels, p, r)

    f1 = 2 * p * r / (p + r)

    return p, r, f1


def tp_pcount_gcount(out, labels):
    """

    :param out:
    :param labels:
    :return: a triple = (true count, predicted_count <tp + fp>, gold_count <tp + fn>)
    """
    outputs = np.argmax(out, axis=1)
    tp = np.sum(np.dot(outputs, labels))
    pred_count = np.sum(outputs == 1)
    gold_count = np.sum(labels == 1)

    return tp, pred_count, gold_count

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "stance": StanceProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model, 
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
