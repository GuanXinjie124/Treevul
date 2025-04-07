这个复刻包包含了我们论文《基于CWE树结构的细粒度提交级漏洞类型预测》的数据集和代码。

我们引入了将检测到的安全补丁分类为细粒度漏洞类型的任务。

我们从NVD收集了一个大型且最新的安全补丁数据集，其中包含来自1,560个GitHub开源软件（OSS）库的6,541个补丁（即我们研究中的提交）。我们根据CWE树的第三层级对补丁进行分类标注。

我们提出了一种名为TreeVul的方法，该方法将CWE树的结构信息作为分类任务的先验知识。📊🔍

环境
操作系统：Ubuntu

GPU：NVIDIA GTX 3090

语言：Python（v3.8）

CUDA：11.2

Python 包：

- PyTorch 1.8.1+cu11
- AllenNLP 2.4.0
- Transformers 4.5.1

请参考这些包的官方文档（特别是AllenNLP）。

设置：

我们修改了Liu等人提出的方法（《即时过时注释检测与更新》，TSE 2021），以提取标记级代码变更信息。他们研究的复刻包已归档在此链接。您可以在这里找到修改后的代码。请注意，我们提出了一种全新的方法来编码代码更改。

我们采用CodeBERT和Bi-LSTM作为神经基线。您可以在这里找到代码。对于Bi-LSTM的训练，我们使用的是GloVe嵌入。请先下载GloVe，然后解压此文件，将glove.6B.300d.txt放入相应文件夹中。

我们使用HuggingFaces Transformer库中的microsoft/codebert-base。您无需自己下载预训练模型，因为在首次运行代码时会自动下载。

根据PyTorch的官方文档，RNN函数（即我们模型中使用的Bi-LSTM组件）存在已知的非确定性问题。我们自己的实验也验证了这一点。因此，遵循该指导，我们通过设置CUDA环境变量CUBLAS_WORKSPACE_CONFIG=:16:8来强制执行确定性行为。为了公平比较（设置CUDA环境也会影响神经网络的其他组件），我们在实验中训练和评估的所有神经网络均使用此CUDA环境。🖥️⚙️✨

数据集

dataset_cleaned.json：收集的来自2,260个GitHub开源软件（OSS）库的10,037个安全补丁（即我们研究中的提交）。请注意，此数据集已经过清理（例如，重复补丁、具有无效CWE的补丁、没有源代码的补丁和较大的补丁）。我们已将相应的CVE ID和CWE ID（以及其路径）与收集的安全补丁结合起来。请注意，此数据集中的每个条目都是一个文件，而分类任务应在提交级别进行。以下是一个示例：
json
{
    "cve_list": "CVE-2010-5332",
    "cwe_list": "CWE-119",
    "path_list": [
        [
            "CWE-664",
            "CWE-118",
            "CWE-119"
        ]
    ],
    "repo": "torvalds/linux",
    "commit_id": "0926f91083f34d047abc74f1ca4fa6a9c161f7db",
    "user": "David S. Miller",
    "commit_date": "2010-10-25T19:14:11Z",
    "msg": "mlx4_en: Fix out of bounds array access\n\nWhen searching for a free entry in either mlx4_register_vlan() or\nmlx4_register_mac(), and there is no free entry, the loop terminates without\nupdating the local variable free thus causing out of array bounds access. Fix\nthis by adding a proper check outside the loop.\n\nSigned-off-by: Eli Cohen <eli@mellanox.co.il>\nSigned-off-by: David S. Miller <davem@davemloft.net>",
    "Total_LOC_REM": 0,
    "Total_LOC_ADD": 11,
    "Total_LOC_MOD": 11,
    "Total_NUM_FILE": 1,
    "Total_NUM_HUNK": 2,
    "file_name": "drivers/net/mlx4/port.c",
    "file_type": "modified",
    "PL": "C",
    "LOC_REM": 0,
    "LOC_ADD": 11,
    "LOC_MOD": 11,
    "NUM_HUNK": 2,
    "REM_DIFF": [
        "",
        ""
    ],
    "ADD_DIFF": [
        "if (free < 0) { err = -ENOMEM; goto out; }",
        "if (free < 0) { err = -ENOMEM; goto out; }"
    ]
}
cve_list 是对应的CVE ID。
cwe_list 是对应的CWE ID。
path_list 是对应CWE的路径（从CWE树的根节点到目标节点）。
repo 是GitHub库的名称。
commit_id 是安全补丁的提交ID。
user 是提交者的名称。
commit_date 是提交日期。
msg 是提交信息。
Total_LOC_ADD、Total_LOC_REM、Total_LOC_MOD 是在整个提交中移除、添加或修改的代码行数（移除和添加的总和）。
Total_NUM_FILE、Total_NUM_HUNK 是此提交中文件和hunk的数量。
以下特征与提交中更改的具体文件相关：

file_name 是更改文件的名称。

PL 是根据文件扩展名推断的编程语言。

LOC_ADD、LOC_REM、LOC_MOD 是在文件中移除、添加或修改的代码行数（移除和添加的总和）。

ADD_DIFF、REM_DIFF 是文件中每个hunk的移除和添加代码段的列表。

dataset_cleaned_level3.json：我们的任务是将安全补丁分类为CWE树第三层级的细粒度类别。我们从dataset_cleaned.json中移除CWE类别为第1层或第2层的补丁。此数据集包含来自1,560个GitHub OSS库的6,541个安全补丁（即我们研究中的提交）。

test_set.json：实验中使用的测试集。我们使用分层随机抽样的方法将dataset_cleaned_level3.json拆分为训练集、验证集和测试集，比例为8:1:1。

train_set.json：实验中使用的训练集。

validation_set.json：实验中使用的验证集。

文件组织

这个项目包含多个文件和三个目录（Baseline_ml - 机器学习基线，Baseline - 深度学习基线，TreeVul - 我们提出的方法及其在消融研究中使用的变体，data - 实验中使用的所有数据）。

文件
predict.py：用于神经模型评估的脚本。您可以使用它评估TreeVul、在消融中使用的变体（即TreeVul-t和TreeVul-h）以及神经基线（即Bi-LSTM和CodeBERT）。我们已经提供了处理这些方法差异的逻辑。

cal_metrics.py：用于计算评估指标的脚本。我们实现了三个包装器，分别用于机器学习模型、没有树结构的神经模型和具有树结构的神经模型。

test_config.json：用于测试神经基线（Bi-LSTM和CodeBERT）及TreeVul-h的配置。这些模型将输入补丁直接映射到其CWE类别。

test_config_tree.json：用于测试TreeVul和TreeVul-t的配置。这两个模型结合了CWE树结构的知识，采用层次和链式模型架构。它们的推断基于我们提出的树结构感知和基于束搜索的推断算法。

utils.py：所有的工具函数，例如将数据集划分为训练集和测试集、构建CWE树、生成CWE路径、预处理数据集。

目录
TreeVul/：TreeVul的代码，包括两个变体，即TreeVul-t（移除标记级代码变更信息）和TreeVul-h（移除层次和链式模型架构设计，用于结合CWE树结构信息）。

reader_treevul_hunk.py：TreeVul的数据集读取器。会调用process_edit.py构建编辑序列token和代码序列token

model_treevul.py：TreeVul和TreeVul-t的模型架构。

config_treevul.json：TreeVul训练的配置。

reader_ablation_noedit_hunk.py：TreeVul-t的数据集读取器。

config_ablation_noedit.json：TreeVul-t训练的配置。

reader_ablation_notree_hunk.py：TreeVul-h的数据集读取器。

model_ablation_notree.py：TreeVul-h的模型架构。

config_ablation_notree.json：TreeVul-h训练的配置。

reader_cwe.py：用于加载CWE描述的数据集读取器，生成标签嵌入。

custom_PTM_embedder.py：自定义嵌入器，在其中添加了对标记级代码变更信息的支持。

custom_trainer.py：自定义训练器，支持自定义回调。

custom_metric.py：自定义验证，计算自定义指标，和cal_metrics.py中的功能相同。

custom_modules.py：构建模型时使用的自定义模块。

callbacks.py：训练中使用的回调，例如，在模型训练前准备CWE树。

process_edit.py：提取标记级代码变更信息。

Baseline/：两个神经基线（Bi-LSTM和CodeBERT）的代码。

tokenizer.py：用于Bi-LSTM的标记器。

reader_baseline_bilstm_hunk.py：Bi-LSTM的数据集读取器。

model_baseline_bilstm.py：Bi-LSTM的模型架构。

config_baseline_bilstm.json：Bi-LSTM训练的配置。

reader_baseline_codebert_hunk.py：CodeBERT的数据集读取器。

model_baseline_codebert.py：CodeBERT的模型架构。

config_baseline_codebert.json：CodeBERT训练的配置。

custom_metric.py：自定义验证，计算自定义指标，与TreeVul中的功能相同。

custom_modules.py：构建模型时使用的自定义模块。

Baseline_ml/：五个机器学习基线的代码。

tokenizer.py：机器学习基线使用的标记器，与Bi-LSTM中使用的相同。

reader_baseline_ml.py：机器学习基线的数据集读取器。

ml_baseline.py：五个机器学习基线的实现，包括随机森林（RF）、线性回归（LR）、支持向量机（SVM）、XGBoost（XGB）和K-最近邻（KNN）。

cal_metrics.py：机器学习基线的评估，和父目录下的功能相同。

data/：实验中使用的所有数据（json格式）。您可以使用utils.py中的相应函数构建这些数据。

cve_data.json：所有的CVE记录。

cwe_tree.json：我们将研究视图中的CWE条目组织成树状结构。

cwe_path.json：将每个CWE类别映射到相应的CWE路径，���从CWE树根节点到目标类别的路径。

valid_cwe_tree.json：仅包含出现在训练集和验证集中的类别的CWE树。

valid_cwes.json：将CWE树每一层的类别映射到索引（仅包括出现在训练集和验证集中的类别）。

训练与测试

要运行机器学习基线模型，请进入 Baseline 目录并运行 python ml_baselines.py（该脚本还会处理评估）。

对于所有神经模型的训练，即 TreeVul 及其变体（TreeVul-t 和 TreeVul-h）和神经基线模型（即 Bi-LSTM 和 CodeBERT）（我们使用 AllenNLP 包实现这些模型），

请在父文件夹中打开终端，并运行 CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train <配置文件> -s <序列化路径> --include-package <包名>。有关更多详细信息，请参阅 AllenNLP 的官方文档。设置 CUDA 环境的原因在“环境”部分进行了说明。

例如，运行 CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train TreeVul/config_treevul.json -s TreeVul/out_treevul/ --include-package TreeVul，您可以在 TreeVul/out_treevul/ 获得输出文件夹以及在控制台中显示的日志信息。
CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train TreeVul/config_treevul.json -s TreeVul/error_treevul/ --include-package TreeVul


对于神经模型的测试，请参阅 predict.py 中的注释。我们已经提供了处理这些方法之间差异的逻辑。您可以运行测试函数以获取每个样本的详细结果（保存于文件 <model>_result.json）和指标（保存于文件 <model>_metric.json）。对于神经模型，您不需要再次使用 cal_metrics.py 来计算指标。

custom_train.py
这是一个用于使用梯度下降进行监督学习的训练器。它只需一个标记的数据集和一个 `DataLoader`，并利用提供的 `Optimizer` 在固定的多个周期内学习模型的权重。你还可以传入一个验证 `data_loader` 并启用早停。还有许多其他的附加功能。

### 注册信息
注册为 `Trainer`，名称为 "gradient_descent"（也是默认的 `Trainer`）。构造函数为 [`from_partial_objects`](#from_partial_objects)，有关该函数的确切键，请查看其参数。如果你使用配置文件，这些参数大致对应于 `__init__` 中的参数，这里不重复它们的文档字符串。

### 参数
- **model** : `Model`, 必需。
    - 要优化的 AllenNLP 模型。如果你使用 GPU 训练模型，它应该已经在正确的设备上。如果使用 AllenNLP 的 `train` 命令，这将为你处理好。

- **optimizer** : `torch.nn.Optimizer`, 必需。
    - 实例化的 PyTorch 优化器，使用要优化的模型参数实例化。

- **data_loader** : `DataLoader`, 必需。
    - 一个包含数据集的 `DataLoader`，返回填充的索引批次。

- **patience** : `Optional[int] > 0`, 可选（默认=`None`）。
    - 在没有改进的情况下，训练将停止的耐心周期数。如果给定，必须大于 0。

- **validation_metric** : `Union[str, List[str]]`, 可选（默认=`"-loss"`）。
    - 用于衡量是否停止训练的验证指标。指标名称必须以 "+" 或 "-" 开头，表示指标是增加还是减少。

- **validation_data_loader** : `DataLoader`, 可选（默认=`None`）。
    - 用于验证集的 `DataLoader`。如果为 None，将使用训练的 `DataLoader` 进行验证。

- **num_epochs** : `int`, 可选（默认=`20`）。
    - 训练周期的数量。

- **serialization_dir** : `str`, 可选（默认=`None`）。
    - 用于保存和加载模型文件的目录路径。如果未传入，则不会保存模型。

- **checkpointer** : `Checkpointer`, 可选（默认=`None`）。
    - 负责定期保存模型权重的 `Checkpointer`。如果未提供，将使用默认参数构造一个。

- **cuda_device** : `int`, 可选（默认=`-1`）。
    - 指定用于该过程的 CUDA 设备。如果为 -1，则使用 CPU。

- **grad_norm** : `float`, 可选（默认=`None`）。
    - 如果提供，将对梯度规范进行缩放，以具有此最大值。

- **grad_clipping** : `float`, 可选（默认=`None`）。
    - 如果提供，梯度将在向后传递期间被剪裁，以具有此最大绝对值。

- **learning_rate_scheduler** : `LearningRateScheduler`, 可选（默认=`None`）。
    - 如果指定，学习率将在每个周期（或批次结束时）根据此计划衰减。

- **momentum_scheduler** : `MomentumScheduler`, 可选（默认=`None`）。
    - 如果指定，动量将在每个批次或周期末按照调度更新。

- **moving_average** : `MovingAverage`, 可选（默认=`None`）。
    - 如果提供，将为所有参数维护移动平均值。

- **callbacks** : `List[Lazy[TrainerCallback]]`, 可选（默认=`None`）。
    - 可以在某些事件（例如每个批次、每个周期、训练开始和结束时）调用的回调列表。

- **distributed** : `bool`, 可选（默认=`False`）。
    - 如果设置，使用 PyTorch 的 `DistributedDataParallel` 在多个GPU上训练模型。

- **local_rank** : `int`, 可选（默认=`0`）。
    - 分布式进程组中 `Trainer` 的唯一标识符。GPU 设备 ID 用作等级。

- **world_size** : `int`, 可选（默认=`1`）。
    - 参与分布式训练的 `Trainer` 工作数量。

- **num_gradient_accumulation_steps** : `int`, 可选（默认=`1`）。
    - 在进行优化器步骤之前累积给定步骤数量的梯度。

- **use_amp** : `bool`, 可选（默认=`False`）。
    - 如果为 `True`，将使用自动混合精度进行训练。

- **enable_default_callbacks** : `bool`, 可选（默认=`True`）。
    - 指示是否在其他回调列表中使用 `DEFAULT_CALLBACKS`。

- **run_sanity_checks** : `bool`, 可选（默认=`True`）。
    - 决定是否运行模型的合理性检查。
CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train TreeVul/config_treevul.json -s TreeVul/useless_out_treevul/ --include-package TreeVul，您可以在 TreeVul/out_treevul/