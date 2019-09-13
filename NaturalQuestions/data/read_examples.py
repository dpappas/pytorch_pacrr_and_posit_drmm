
import numpy as np
import json
import base64
from pprint import pprint
import gzip
from tqdm import tqdm

def has_long_answer(json_example):
  for annotation in json_example['annotations']:
    if annotation['long_answer']['start_byte'] >= 0:
      return True
  return False

def has_short_answer(json_example):
  for annotation in json_example['annotations']:
    if annotation['short_answers']:
      return True
  return False

class LongAnswerCandidate(object):
  """Representation of long answer candidate."""
  ################################################
  def __init__(self, contents, index, is_answer, contains_answer):
    self.contents = contents
    self.index = index
    self.is_answer = is_answer
    self.contains_answer = contains_answer
    if is_answer:
      self.style = 'is_answer'
    elif contains_answer:
      self.style = 'contains_answer'
    else:
      self.style = 'not_answer'

class Example(object):
  """Example representation."""
  def __init__(self, json_example, dataset):
    self.json_example = json_example
    # Whole example info.
    self.url = json_example['document_url']
    self.title = (
        json_example['document_title']
        if json_example.has_key('document_title') else 'Wikipedia')
    self.example_id = base64.urlsafe_b64encode(
        str(self.json_example['example_id']))
    self.document_html = self.json_example['document_html'].encode('utf-8')
    self.document_tokens = self.json_example['document_tokens']
    self.question_text = json_example['question_text']
    if dataset == 'train':
      if len(json_example['annotations']) != 1:
        raise ValueError(
            'Train set json_examples should have a single annotation.')
      annotation = json_example['annotations'][0]
      self.has_long_answer = annotation['long_answer']['start_byte'] >= 0
      self.has_short_answer = annotation[
          'short_answers'] or annotation['yes_no_answer'] != 'NONE'
    elif dataset == 'dev':
      if len(json_example['annotations']) != 5:
        raise ValueError('Dev set json_examples should have five annotations.')
      self.has_long_answer = sum([
          annotation['long_answer']['start_byte'] >= 0
          for annotation in json_example['annotations']
      ]) >= 2
      self.has_short_answer = sum([
          bool(annotation['short_answers']) or
          annotation['yes_no_answer'] != 'NONE'
          for annotation in json_example['annotations']
      ]) >= 2
    self.long_answers = [
        a['long_answer']
        for a in json_example['annotations']
        if a['long_answer']['start_byte'] >= 0 and self.has_long_answer
    ]
    self.short_answers = [
        a['short_answers']
        for a in json_example['annotations']
        if a['short_answers'] and self.has_short_answer
    ]
    self.yes_no_answers = [
        a['yes_no_answer']
        for a in json_example['annotations']
        if a['yes_no_answer'] != 'NONE' and self.has_short_answer
    ]
    if self.has_long_answer:
      long_answer_bounds = [
          (la['start_byte'], la['end_byte']) for la in self.long_answers
      ]
      long_answer_counts = [
          long_answer_bounds.count(la) for la in long_answer_bounds
      ]
      long_answer = self.long_answers[np.argmax(long_answer_counts)]
      self.long_answer_text = self.render_long_answer(long_answer)
    else:
      self.long_answer_text = ''
    if self.has_short_answer:
      short_answers_ids = [[
          (s['start_byte'], s['end_byte']) for s in a
      ] for a in self.short_answers] + [a for a in self.yes_no_answers]
      short_answers_counts = [
          short_answers_ids.count(a) for a in short_answers_ids
      ]
      self.short_answers_texts = [
          ', '.join([
              self.render_span(s['start_byte'], s['end_byte'])
              for s in short_answer
          ])
          for short_answer in self.short_answers
      ]
      self.short_answers_texts += self.yes_no_answers
      self.short_answers_text = self.short_answers_texts[np.argmax(short_answers_counts)]
      self.short_answers_texts = set(self.short_answers_texts)
    else:
      self.short_answers_texts = []
      self.short_answers_text = ''
    self.candidates = self.get_candidates(
        self.json_example['long_answer_candidates'])
    self.candidates_with_answer = [
        i for i, c in enumerate(self.candidates) if c.contains_answer
    ]
  ######################################
  def render_long_answer(self, long_answer):
    """Wrap table rows and list items, and render the long answer.

    Args:
      long_answer: Long answer dictionary.

    Returns:
      String representation of the long answer span.
    """
    if long_answer['end_token'] - long_answer['start_token'] > 500:
      return 'Large long answer'
    html_tag = self.document_tokens[long_answer['end_token'] - 1]['token']
    if html_tag == '</Table>' and self.render_span(
        long_answer['start_byte'], long_answer['end_byte']).count('<TR>') > 30:
      return 'Large table long answer'
    elif html_tag == '</Tr>':
      return '<TABLE>{}</TABLE>'.format(
          self.render_span(long_answer['start_byte'], long_answer['end_byte']))
    elif html_tag in ['</Li>', '</Dd>', '</Dd>']:
      return '<Ul>{}</Ul>'.format(
          self.render_span(long_answer['start_byte'], long_answer['end_byte']))
    else:
      return self.render_span(long_answer['start_byte'], long_answer['end_byte'])
  ######################################
  def render_span(self, start, end):
    return self.document_html[start:end]
  ######################################
  def get_candidates(self, json_candidates):
    """Returns a list of `LongAnswerCandidate` objects for top level candidates.
    Args:
      json_candidates: List of Json records representing candidates.
    Returns:
      List of `LongAnswerCandidate` objects.
    """
    candidates = []
    top_level_candidates = [c for c in json_candidates if c['top_level']]
    for candidate in top_level_candidates:
      tokenized_contents = ' '.join([
          t['token'] for t in self.json_example['document_tokens']
          [candidate['start_token']:candidate['end_token']]
      ])
      start = candidate['start_byte']
      end = candidate['end_byte']
      is_answer = self.has_long_answer and np.any(
          [(start == ans['start_byte']) and (end == ans['end_byte'])
           for ans in self.long_answers])
      contains_answer = self.has_long_answer and np.any(
          [(start <= ans['start_byte']) and (end >= ans['end_byte'])
           for ans in self.long_answers])
      candidates.append(LongAnswerCandidate(tokenized_contents, len(candidates), is_answer, contains_answer))
    return candidates

def _load(examples, f, dataset, total):
    """Read serialized json from `f`, create examples, and add to `examples`."""
    for l in tqdm(f, total=total):
        json_example = json.loads(l)
        if not has_long_answer(json_example) or not has_short_answer(json_example):
            continue
        # pprint(json_example)
        example = Example(json_example, dataset=dataset)
        examples[example.example_id] = example
        # if len(examples) == FLAGS.max_examples:
        #     break

examples = {}
with open('/media/dpappas/dpappas_data/NaturalQuestions/v1.0_sample_nq-train-sample.jsonl.gz') as fileobj:
    _load(examples, gzip.GzipFile(fileobj=fileobj), 'train', 200)

examples = {}
with open('/media/dpappas/dpappas_data/NaturalQuestions/v1.0_sample_nq-dev-sample.jsonl.gz') as fileobj:
    _load(examples, gzip.GzipFile(fileobj=fileobj), 'dev', 200)

