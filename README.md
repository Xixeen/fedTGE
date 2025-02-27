 # ENERGY-BASED BACKDOOR DEFENSE AGAINST FED-ERATED GRAPH LEARNING

> Guancheng Wan, Zitong Shi, Wenke Huang, Guibin Zhang, Dacheng Tao, Mang Ye


## Abstract

Federated Graph Learning is rapidly evolving as a privacy-preserving collaborative approach. However, backdoor attacks are increasingly undermining federated systems by injecting carefully designed triggers that lead the model making incorrect predictions. Trigger structures and injection locations in Federated Graph Learning are more diverse, making traditional federated defense methods less effective. In our work, we propose an effective Federated Graph Backdoor Defense using Topological Graph Energy (FedTGE). At the client level, it injects distribution knowledge into the local model, assigning low energy to benign samples and high energy to the constructed malicious substitutes, and selects benign clients through clustering. At the server level, the energy elements uploaded by each client are treated as new nodes to construct a global energy graph for energy propagation, making the selected clients’ energy elements more similar and further adjusting the aggregation weights. Our method can handle high data heterogeneity, does not require a validation dataset, and is effective under both small and large malicious proportions. Extensive results on various settings of federated graph scenarios under backdoor attacks validate the effectiveness of this approach. 


## Citation
```
@inproceedings{FedTGE_ICLR25,
  title={Energy-based Backdoor Defense Against Federated Graph Learning},
  author={Wan, Guancheng and Shi, Zitong and Huang, Wenke and Zhang, Guibin and Tao, Dacheng and Ye, Mang},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

