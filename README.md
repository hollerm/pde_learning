# Learning-informed parameter identification in nonlinear time-dependent PDEs
This repository provides the source code for the paper "Learning-informed parameter identification in nonlinear time-dependent PDEs" as cited below.

 
## Requirements
The code was written and tested with Python 3.10.10 under Linux. No dedicated installation is needed for the code, simply download the code and get started. Be sure to have the following Python modules installed, most of which should be standard.

* [numpy], tested with version 1.24.3
* [scipy], tested with version 1.10.1
* [matplotlib], testd with version 3.7.1
* [torch], tested with version 2.0.0
* [time]
* [sys]
* [copyreg]
* [types]
* [pickle]
* [os]
* [itertools]

We recomment to use torch.cuda to get GPU support.

## Examples

* To run a quick demo example, call "python quick_demo.py" in a terminal. This should compute results for a simple experiment and store them in the folder "demo_results".

* To re-compute experiments of the paper, call "reproduce_experiments.py". This will compute all results of the paper that were computed with the PyTorch code (it might take a while). To select only specific results, see the source code of "reproduce_experiments.py" and select only specific experiments.

## Authors of the code

* **Christian Aarset** c.aarset@math.uni-goettingen.de
* **Martin Holler** martin.holler@uni-graz.at 
* **Tram Thi Ngoc Nguyen** nguyen@mps.mpg.de

CA is currently affiliated with the Institute for Numerical and Applied Mathematics, University of Göttingen, Göttingen, Germany. MH is currently affiliated with the Department of Mathematics and Scientific Computing, University of Graz, Graz, Austria. TTNN is currently affiliated with the Max Planck Institute for Solar System Research, Göttingen, Germany.

## Publications
If you use this code, please cite the following associated publication.

* Aarset, C., Holler, M., Nguyen, T.T.N. Learning-Informed Parameter Identification in Nonlinear Time-Dependent PDEs. Applied Mathematics & Optimization, 2023. https://doi.org/10.1007/s00245-023-10044-y

## License

The code in this project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
