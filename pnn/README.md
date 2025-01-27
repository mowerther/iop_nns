## `pnn` module

This module has the following components:
* [`pnn.nn`](nn): 
Neural network architectures and functionality relating to file input/output, training, testing, estimation, etc.

* [`pnn.output`](output): 
Scientific outputs, such as drawing figures, writing out tables, etc.

* `pnn.aggregate`: 
Helper functions for aggregating results, e.g. calculating MdSA and other metrics for each scenario-architecture combination.

* `pnn.constants`: 
Constants used elsewhere in the code, such as default file locations and parameters.
Provides the `Parameter` dataclass which ensures consistent nomenclature, units, colours, etc. for scenarios, IOPs, uncertainty types, and metrics.

* `pnn.data`: 
Reading and pre-processing input data (original and split) as well as re-scaling of IOPs.

* `pnn.maps`:
Applying neutral network modules to PRISMA scenes and plotting the results.

* `pnn.metrics`:
Calculating metrics like MdSA, SSPB, R², coverage, etc.

* `pnn.modeloutput`: 
Reading and pre-processing model outputs (IOP estimates).

* `pnn.recalibration`: 
Neural network recalibration; fitting and applying recalibration functions, calculating calibration curves.

* `pnn.split`: 
Splitting data.
