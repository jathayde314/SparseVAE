## Sparse VAE
### Course Final Project

Project Report link: [final_project.pdf](final_project.pdf)

### Instructions
1. To train the SparseVAE
```
python sparsevae.py --data-folder /tmp/miniimagenet --output-folder models/sparsevae
```

### Overview
This project extends the VAE framework by attempting to train a Sparse Dictionary between 

### Comment
The method developed here has notable drawbacks in that it enforces limited sparsity and does not improve performance while requiring an extra order `d` more runtime to iterate over each dimension of the latent space. For a better way to implement a Sparse VAE, refer to https://arxiv.org/abs/2110.10804. Note that this paper was published after I submitted this project.

### Authors
Joshua Athayde

### Acknowledgements
I would like to thank Rithesh Kumar, Tristan Deleu, and Evan Racah for reproducing the VAE framework in Pytorch and making it publicly available. The work in this repository extends their code to attempt this new task.

