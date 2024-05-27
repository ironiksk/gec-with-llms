
# Enhancing Grammatical Correctness: The Efficacy of Large Language Models in Error Correction Task

This repository provides code for results reproducing, predictions and links to the pretrained Grammatical Error Correction models for Master Thesis "Enhancing Grammatical Correctness: The Efficacy of Large Language Models in Error Correction Task". 


This work was done as a part of the research <a href="https://arxiv.org/abs/2404.14914">"Pillars of Grammatical Error Correction: Comprehensive Inspection Of Contemporary Approaches In The Era of Large Language Models"</a>.


# Structure

* `finetuning` directory contain required code to reproduce Full and PERF finetuning setup. We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main) tool for model finetuning.

* `data` directory contain systems prediction on 3 main GEC benchmarks.

* `metrics` directory contain evaluation scripts used in the work. 

* `sampling` directory contains helper scrtips and notebooks for corrected texts sampling with pretrained model for DPO experiments.

* `zero-shot` directory contains notebooks for prompts evaluation in zero-shot setup.


## Pretrained models and results

Table bellow contain single system scores and links to trained models available for download.  

<table>
  <tr>
    <th>Model name</th>
    <th colspan="3">CoNNL-2014 (test)</th>
    <th colspan="3">BEA-2019 (dev)</th>
    <th colspan="3">BEA-2019 (test)</th>
  </tr>
  <tr>
    <th>  </th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F05</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F05</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F05</th>
  </tr>

  <tr>
    <td>Chat-LLaMa-2-7B-FT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/llama7b.tar">[link]</a></td>
    <td>75.5</td>
    <td>46.8</td>
    <td>67.2</td> 
    <td>58.3</td>
    <td>46.0</td>
    <td>55.3</td> 
    <td>72.3</td>
    <td>67.4</td>
    <td>71.2</td> 
  </tr>
  <tr>
    <td>Chat-LLaMa-2-13B-FT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/llama13b.tar">[link]</a></td>
    <td>77.2</td>
    <td>45.6</td>
    <td>67.9</td> 
    <td>59.8</td>
    <td>46.1</td>
    <td>56.4</td> 
    <td>74.6</td>
    <td>67.8</td>
    <td>73.1</td> 
  </tr>

</table>


## Evaluation and Metrics

There are 3 evaluation sets that we are using for GEC:

1. CoNLL-2014 (`nucle14-2a`, m2 file is available; [m2scorer](https://gitlab.grammarly.io/nlp-research/m2scorer) is official scorer)
2. BEA19-dev (`bea-dev`, m2 file is available; [errant](https://github.com/chrisjbryant/errant) is official scorer and it's dockerized version in directory `metrics`)
3. BEA19-test (`bea-test`, m2 file is NOT available; score can be got only through [codelab](https://codalab.lisn.upsaclay.fr/competitions/4057#results
) sumbission)

### Examples of evaluation

Evalsest directory: `data/evaluation_sets`.

1. Example of evaluation with Errant

```
ERRANT_SCORER=path_to_errant_scorer_directory
INPUT_FILE=data/evaluation_sets/bea-dev.txt
M2_FILE=data/evaluation_sets/bea-dev.m2
PRED_FILE=YOUR_PRED_FILE.txt
TMP_FILE=YOUR_TMP_FILE.m2


python $ERRANT_SCORER/parallel_to_m2.py -orig $INPUT_FILE -cor $PRED_FILE -out $TMP_FILE
python $ERRANT_SCORER/compare_m2.py -hyp $TMP_FILE -ref $M2_FILE >> {{result}}
```


2. Example of evaluation with m2scorer
```
M2_SCORER=path_to_m2scorer
M2_FILE=data/evaluation_sets/nucle14-2a.m2
PRED_FILE=YOUR_PRED_FILE.txt
$M2_SCORER $PRED_FILE $M2_FILE >> {{reslut}}
```

## Citation
[to be updated once proceedings are published]


## References
```
@misc{omelianchuk2024pillars,
      title={Pillars of Grammatical Error Correction: Comprehensive Inspection Of Contemporary Approaches In The Era of Large Language Models}, 
      author={Kostiantyn Omelianchuk and Andrii Liubonko and Oleksandr Skurzhanskyi and Artem Chernodub and Oleksandr Korniienko and Igor Samokhin},
      year={2024},
      eprint={2404.14914},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement

During this challenging time when Ukraine continues to resist the unprovoked Russian invasion, I am thankful to our communities and to everyone who defends our homeland, supports our people, and sends aid amidst the ongoing conflict.

I am grateful to my supervisor, Kostiantyn Omelianchuk, Grammarly and the team for guidance that enabled this research.

I am also grateful to the Ukrainian Catholic University for their outstanding master’s program.

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main), [vLLM](https://github.com/vllm-project/vllm). 
