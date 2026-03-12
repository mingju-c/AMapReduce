<div align="center">

<h2>A-MapReduce: Executing Wide Search via Agentic MapReduce</h2>

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.01331-b31b1b.svg)](https://arxiv.org/abs/2602.01331)

</div>
<div align="center">
    <img src="./assets/A-MapReduce.png" width="100%" height="auto" />
</div>


## Introduction
This is the official repository for **A-MapReduce**, a MapReduce-style multi agent framework that achieves SOTA results on wide search datasets.

Unlike deep-research settings that emphasize vertically recursive reasoning, wide-search tasks require agents to discover, track, and aggregate a **large set of weakly coupled retrieval targets** under long-horizon execution. Existing agentic systems are still largely built around sequential planning and implicit target management, which often leads to missing entries, redundant retrieval, and inefficient execution at scale.

A-MapReduce addresses this by reformulating wide search as a **horizontally structured retrieval problem**. Given a query, the framework constructs an explicit **MapReduce-style execution decision**, decomposes the task into atomic retrieval units, executes them in parallel batches, and reduces intermediate results into a unified schema-consistent table. This makes large-scale retrieval objectives more controllable, traceable, and efficient. Beyond structured execution, A-MapReduce further introduces an **experiential memory** mechanism that distills historical execution trajectories into reusable structural hints. By retrieving and reusing these experiences, the framework progressively improves task decomposition, allocation, and recomposition over time, enabling more adaptive, robust, and cost-effective wide-search execution.



## Quick Start ⚙
### 1. Env Setup
```bash
conda create -n a_mapreduce python=3.10
conda activate a_mapreduce
pip install -r requirements.txt
pip install langchain_chroma==1.0.0 langchain-core==1.0.4 gradio==5.22.0 aiofiles==23.2.1
```

### 2. Set up environment variables

A-MapReduce framework and model use `SearchTool` and `CrawlTool` for web search and crawl pages, which require environment variables with the corresponding API key, based on the selected provider:
- `SERPER_API_KEY` for SerpApi: [Serper]
(https://serper.dev/)
- `JINA_API_KEY` for JinaApi: [JinaAI]
(https://jina.ai/)

Depending on the model you want to use, you may need to set environment variables. You need to set the `DEFAULT_MODEL`, `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variable.

### 3. Run A-Mapreduce Framework
#### Step 1: Cold start (no plan mode)

Widesearch:
```bash
python run_widesearch.py --enable_expmemory --expmemory_namespace widesearch --selected_tasks <task_index_1> <task_index_2> ... --trial_num 4
```
DeepWideSearch:
```bash
python run_deepwidesearch.py --enable_expmemory --expmemory_namespace deepwidesearch --selected_tasks <task_index_1> <task_index_2> ... --trial_num 14
```
Note: Cold start is complete once the memory store has accumulated enough retrievable tasks and `insights.json` contains usable hints (you should see multiple non-empty entries). You can control the hint generation pace by tuning `--expmemory_start_insights_threshold` and `--expmemory_rounds_per_insights`. `--selected_tasks` uses dataset indices (0-based) from the input JSON/JSONL file; for example, `--selected_tasks 0 3 5`.

#### Step 2: Normal self-evolution (plan mode)

Widesearch:
```bash
python run_widesearch.py --enable_expmemory --expmemory_namespace widesearch --selected_tasks <task_index_1> <task_index_2> ... --trial_num 4 --mapreduce_plan_mode
```
DeepWideSearch:
```bash
python run_deepwidesearch.py --enable_expmemory --expmemory_namespace deepwidesearch --selected_tasks <task_index_1> <task_index_2> ... --trial_num 4 --mapreduce_plan_mode
```

### 4. Quick demo of self-evolution (using pre-warmed memory)
Widesearch example:
```bash
python run_widesearch.py --enable_expmemory --expmemory_namespace "widesearch_test" --mapreduce_plan_mode --selected_tasks 12 --trial_num 4
```
DeepWideSearch example :
```bash
python run_deepwidesearch.py --enable_expmemory --expmemory_namespace "deepwidesearch_test" --mapreduce_plan_mode --selected_tasks 41 --trial_num 4
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--max_steps` | Maximum agent steps | `40` |
| `--mapreduce_max_steps` | Max steps for sub-agents spawned by mapreducetool | `40` |
| `--mapreduce_plan_mode` | Enable plan+execute two-phase mode | `store_true`  |
| `--mapreduce_insight_topk` | Number of memory hints to include in mapreduce plan stage | `3` |
| `--selected_tasks` | Optional list of dataset indices to run | `0 1 2` |
| `--trial_num` | Number of trials per task | `4` |
| `--enable_expmemory` | Enable record logging | `store_true` |
| `--expmemory_start_insights_threshold` | Minimum memory size before insights updates start | `40` |
| `--expmemory_rounds_per_insights` | Update insights every N new records | `20` |
| `--expmemory_merge_insights_interval` | Merge insights every N records once threshold is reached | `40` |

## Citation

If you find A-MapReduce helpful in your research, please kindly consider citing:

```bibtex
@misc{chen2026amapreduceexecutingwidesearch,
      title={A-MapReduce: Executing Wide Search via Agentic MapReduce}, 
      author={Mingju Chen and Guibin Zhang and Heng Chang and Yuchen Guo and Shiji Zhou},
      year={2026},
      eprint={2602.01331},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2602.01331}, 
}
```

---

## Acknowledgments

This work builds upon and adapts code from:

- **[Flash-Searcher](https://github.com/OPPO-PersonalAI/Flash-Searcher)** — DAG-based parallel execution framework for agent task execution and trajectory collection

We sincerely thank the contributors of these projects for their excellent work in advancing agent-based systems.

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
