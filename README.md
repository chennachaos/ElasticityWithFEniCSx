# ElasticityWithFEniCSx

This repository contains source files (Python scripts) for solving elasticity problems using FEniCSx library, the new version of FEniCS.

The input meshes are generated using **GMSH** and used directlyy using the function `gmshio.read_from_msh()` instead of converting to the **XML** format.

Some of the example problems solved are shown below.

## Linear and Hyperelasticity
### Linear elasticity
#### Cook's membrane in plane-strain condition
<img src="./LinearAndHyperelastic/Cooksmembrane2D-LE/Cooksmembrane2d-LE-nelem8-dispY.png" alt="Y-displacement" width="250"/>
<img src="./LinearAndHyperelastic/Cooksmembrane2D-LE/Cooksmembrane2d-LE-nelem8-pressure.png" alt="Pressure" width="250"/>


### Hyperelasticity

#### Cook's membrane in plane-strain condition
<img src="./LinearAndHyperelastic/Cooksmembrane2D-NH/Cooksmembrane2d-NH-nelem8-dispY.png" alt="Y-displacement" width="250"/>
<img src="./LinearAndHyperelastic/Cooksmembrane2D-NH/Cooksmembrane2d-NH-nelem8-pressure.png" alt="Pressure" width="250"/>

#### Cook's membrane in 3D
<img src="./LinearAndHyperelastic/Cooksmembrane3D-NH/Cooksmembrane3d-NH-nelem4-dispY.png" alt="Y-displacement" width="250"/>
<img src="./LinearAndHyperelastic/Cooksmembrane3D-NH/Cooksmembrane3d-NH-nelem4-pressure.png" alt="Pressure" width="250"/>


#### Block 3D
<img src="./LinearAndHyperelastic/Block3d/block3d-nelem4-dispZ.png" alt="Z-displacement" width="250"/>


## Morphoelasticity - Growth-driven deformations

#### Dilatation of a Cube
<img src="./Growth/Cube/cube-growth-10times-dispX.png" alt="Growth Cube" width="250"/>


#### Deformation of rods
<img src="./Growth/Rod/rod3d-circle.png" alt="Rod - Circle shape" width="400"/>

## Magnetomechanics

#### Softmagnetic beam
<img src="./Magneto-Soft/beam3d/SoftMagnetic-beam3D-phi.png" alt="Rod - Circle shape" width="400"/>


