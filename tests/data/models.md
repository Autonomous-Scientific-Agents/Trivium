# Benchmarking Interatomic Potentials

## Introduction

MACE is a machine-learned interatomic potential. XTB is a semi-empirical tight-binding method. ANI is a neural network potential. MACE was developed by Cambridge Group. XTB was developed by Grimme Group. ANI was developed by Roitberg Group.

## Methods

MACE was trained on QM9X. ANI was trained on ANI-1x. The simulation temperature of Run 2 was 450 K. The simulation temperature of Run 3 was 600 K. The cutoff radius of MACE was 5.0 Angstrom. MACE uses equivariant message passing. ANI uses atomic environment vectors.

## Results

MACE achieved a mean absolute error of 0.12 eV. XTB achieved a mean absolute error of 0.21 eV. ANI achieved a mean absolute error of 0.35 eV. MACE achieved a training time of 14 hours. ANI achieved a training time of 9 hours. XTB achieved a runtime of 3 seconds. MACE achieved a runtime of 40 seconds. The initial error of MACE was 0.31 eV. The final error of MACE was 0.18 eV.

## Conclusion

MACE has a parameter count of 2.4 million. ANI has a parameter count of 0.9 million. Cambridge Group was founded in 1209. Grimme Group was founded in 2006. Cambridge Group is located in Cambridge.
