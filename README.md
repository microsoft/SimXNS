# SimXNS

[✨Updates](#️Updates) | [📜Citation](#Citation) | [🤘Furthermore](#Furthermore) | [❤️Contributing](#Contributing) | [📚Trademarks](#Trademarks)

[SimXNS](https://aka.ms/simxns) is a research project for information retrieval by MSRA NLC team. Some of the techniques are actively used in [Microsoft Bing](https://www.bing.com/). This repo provides the official code implementations.

Currently, this repo contains several methods that are designed for or related to information retrieval.
Here are some basic descriptions to help you catch up with the characteristics of each work:
- [**SimANS**](https://arxiv.org/abs/2210.11773) is a simple, general and flexible ambiguous negatives sampling method for dense text retrieval. It can be easily applied to various dense retrieval methods like [AR2](https://github.com/microsoft/AR2). This method is also applied in [Bing](https://www.bing.com/) search engine, which is proven to be effective.
- [**MASTER**](https://arxiv.org/abs/2212.07841) is a multi-task pre-trained model that unifies and integrates multiple pre-training tasks with different learning objectives under the bottlenecked masked autoencoder architecture.
- [**PROD**](https://arxiv.org/abs/2209.13335) is a novel distillation framework for dense retrieval, which consists of a teacher progressive distillation and a data progressive distillation to gradually improve the student.
- [**CAPSTONE**](https://arxiv.org/abs/2212.09114) is a curriculum sampling for dense retrieval with document expansion, to bridge the gap between training and inference for dual-cross-encoder.
- [**ALLIES**](https://arxiv.org/abs/2305.14766) leverages LLMs to iteratively generate new queries related to the original query, enabling an iterative reasoning process. By iteratively refining and expanding the scope of the original query, ALLIES captures and utilizes hidden knowledge that may not be directly obtainable through retrieval.
- [**LEAD**](https://arxiv.org/abs/2212.05225) aligns the layer features of student and teacher, emphasizing more on the informative layers by re-weighting.


## Updates

- 2023/10/29: release the official code of [CAPSTONE](https://github.com/microsoft/SimXNS/tree/main/CAPSTONE).
- 2023/10/18: release the official code of [ALLIES](https://github.com/microsoft/SimXNS/tree/main/ALLIES).
- 2023/07/03: upload the pretrained MASTER checkpoints for [MARCO](https://huggingface.co/lx865712528/master-base-pretrained-msmarco) and [Wikipedia](https://huggingface.co/lx865712528/master-base-pretrained-wiki) to huggingface model hub.
- 2023/07/03: update approaches for downloading resources.
- 2023/05/29: release the official code of [LEAD](https://github.com/microsoft/SimXNS/tree/main/LEAD).
- 2023/02/16: refine the resources of [SimANS](https://github.com/microsoft/SimXNS/tree/main/SimANS) by uploading files in a seperated style and offering the [file list](https://github.com/microsoft/SimXNS/tree/main/SimANS#-how-to-use).
- 2023/02/02: release the official code of [PROD](https://github.com/microsoft/SimXNS/tree/main/PROD).
- 2022/12/16: release the official code of [MASTER](https://github.com/microsoft/SimXNS/tree/main/MASTER).
- 2022/11/17: release the official code of [SimANS](https://github.com/microsoft/SimXNS/tree/main/SimANS).


## Citation
If you extend or use this work, please cite our paper where it was introduced:


- **SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval**. Kun Zhou, Yeyun Gong, Xiao Liu, Wayne Xin Zhao, Yelong Shen, Anlei Dong, Jingwen Lu, Rangan Majumder, Ji-Rong Wen, Nan Duan, Weizhu Chen. ***EMNLP Industry Track 2022***. [Code](https://github.com/microsoft/SimXNS/tree/main/SimANS), [Paper](https://arxiv.org/abs/2210.11773).
- **MASTER: Multi-task Pre-trained Bottlenecked Masked Autoencoders are Better Dense Retrievers**. Kun Zhou, Xiao Liu, Yeyun Gong, Wayne Xin Zhao, Daxin Jiang, Nan Duan, Ji-Rong Wen. ***ECML-PKDD 2023***. [Code](https://github.com/microsoft/SimXNS/tree/main/MASTER), [Paper](https://arxiv.org/abs/2212.07841).
- **PROD: Progressive Distillation for Dense Retrieval**. Zhenghao Lin, Yeyun Gong, Xiao Liu, Hang Zhang, Chen Lin, Anlei Dong, Jian Jiao, Jingwen Lu, Daxin Jiang, Rangan Majumder, Nan Duan. ***WWW 2023***. [Code](https://github.com/microsoft/SimXNS/tree/main/PROD), [Paper](https://arxiv.org/abs/2209.13335).
- **CAPSTONE: Curriculum Sampling for Dense Retrieval with Document Expansion**. Xingwei He, Yeyun Gong, A-Long Jin, Hang Zhang, Anlei Dong, Jian Jiao, Siu Ming Yiu, Nan Duan. ***EMNLP 2023***. [Code](https://github.com/microsoft/SimXNS/tree/main/CAPSTONE), [Paper](https://arxiv.org/abs/2212.09114).
- **Allies: Prompting Large Language Model with Beam Search**. Hao Sun, Xiao Liu, Yeyun Gong, Yan Zhang, Daxin Jiang, Linjun Yang, Nan Duan. ***Findings of EMNLP 2023***. [Code](https://github.com/microsoft/SimXNS/tree/main/ALLIES), [Paper](https://arxiv.org/abs/2305.14766).
- **LEAD: Liberal Feature-based Distillation for Dense Retrieval**. Hao Sun, Xiao Liu, Yeyun Gong, Anlei Dong, Jian Jiao, Jingwen Lu, Yan Zhang, Daxin Jiang, Linjun Yang, Rangan Majumder, Nan Duan. ***WSDM 2024***. [Code](https://github.com/microsoft/SimXNS/tree/main/LEAD), [Paper](https://arxiv.org/abs/2212.05225).


```bibtex
@inproceedings{zhou2022simans,
   title     = {{SimANS:} Simple Ambiguous Negatives Sampling for Dense Text Retrieval},
   author    = {Kun Zhou, Yeyun Gong, Xiao Liu, Wayne Xin Zhao, Yelong Shen, Anlei Dong, Jingwen Lu, Rangan Majumder, Ji-Rong Wen, Nan Duan and Weizhu Chen},
   booktitle = {{EMNLP Industry Track}},
   year      = {2022}
}
@inproceedings{zhou2023master,
   title     = {{MASTER:} Multi-task Pre-trained Bottlenecked Masked Autoencoders are Better Dense Retrievers},
   author    = {Kun Zhou, Xiao Liu, Yeyun Gong, Wayne Xin Zhao, Daxin Jiang, Nan Duan, Ji-Rong Wen},
   booktitle = {{ECML-PKDD}},
   year      = {2023}
}
@inproceedings{lin2023prod,
   title     = {{PROD:} Progressive Distillation for Dense Retrieval},
   author    = {Zhenghao Lin, Yeyun Gong, Xiao Liu, Hang Zhang, Chen Lin, Anlei Dong, Jian Jiao, Jingwen Lu, Daxin Jiang, Rangan Majumder and Nan Duan},
   booktitle = {{WWW}},
   year      = {2023}
}
@inproceedings{he2023capstone,
   title     = {{CAPSTONE:} Curriculum Sampling for Dense Retrieval with Document Expansion},
   author    = {Xingwei He, Yeyun Gong, A-Long Jin, Hang Zhang, Anlei Dong, Jian Jiao, Siu Ming Yiu and Nan Duan},
   booktitle = {{EMNLP}},
   year      = {2023}
}
@inproceedings{sun2023allies,
   title     = {{Allies:} Prompting Large Language Model with Beam Search},
   author    = {Hao Sun, Xiao Liu, Yeyun Gong, Yan Zhang, Daxin Jiang, Linjun Yang and Nan Duan},
   booktitle = {{Findings of EMNLP}},
   year      = {2023}
}
@inproceedings{sun2024lead,
   title     = {{LEAD:} Liberal Feature-based Distillation for Dense Retrieval},
   author    = {Hao Sun, Xiao Liu, Yeyun Gong, Anlei Dong, Jian Jiao, Jingwen Lu, Yan Zhang, Daxin Jiang, Linjun Yang, Rangan Majumder and Nan Duan},
   booktitle = {{WSDM}},
   year      = {2024}
}
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
