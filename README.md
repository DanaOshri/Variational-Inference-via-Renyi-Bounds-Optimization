# Variational-Inference-via-Renyi-Bounds-Optimization
This project contains code for our paper "Variational Inference via Rényi Upper-Lower Bound Optimization".

<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="@inproceedings{VRLU2022OshriFine,
  title={Variational Inference via Rényi Upper-Lower Bound Optimization},
  author={Dana Oshri Zalman, Shai Fine},
  booktitle={International Conference on Machine Learning and Applications (ICMLA)},
  year={2022}
}"><pre class="notranslate"><code>@inproceedings{VRLU2022OshriFine,
  title={Variational Inference via Rényi Upper-Lower Bound Optimization},
  author={Dana Oshri Zalman, Shai Fine},
  booktitle={International Conference on Machine Learning and Applications (ICMLA)},
  year={2022}
}
</code></pre></div>

It contains pytorch implementation of VAE, using different loss functions:
  1. Maximizing the ELBO.
  2. Maximizing Rényi Lower Bound with positive alpha.
  3. Minimizing Rényi Upper Bound with negative alpha.
  4. Using Rényi Upper-Lower Bounds combination as the loss function.

We used 3 datasets: MNIST, USPS and SVHN.
