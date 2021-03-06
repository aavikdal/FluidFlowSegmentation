# FluidFlowSegmentation
These repository contains scripts that accompany the master thesis Vikdal (2021).

**Master Project**

The *morphologicalInvasion.py* script is written by Carl Fredrik Berg and made in the thesis Berg et.al (2020). Some parts of the original script has been removed as they were not needed, and some minor adjustments to fit the packing data in this project has been made.

The *ReadPackingScript.m* is from Baranau and Tallarek (2014) and is also available on this [Github page](https://github.com/VasiliBaranov/packing-generation), along with the rest of the Packing Generation Project.

The *multiResUNet2D.py* is a slight modification of the original implementaton found in this [Github page](https://github.com/nibtehaz/MultiResUNet), and published in  Ibtehaz and Rahman (2020).

**Specialization Project**

The implementation of U-Net and its training process is inspired and influenced by this [U-Net tutorial](https://github.com/nikhilroxtomar/Multiclass-Segmentation-in-Unet). This includes *UNet.py*, *prepare_data_unet.py* and *train_unet.py*.

The feature extraction process implemented in *feature_extraction.py* is influenced by this [Python for microsoft tutorial series](https://github.com/bnsreenu/python_for_microscopists).

References:

Baranau, V., Tallarek, U. (2020). Random-close packing limits for monodisperspe and polydisperse hard spheres. *Soft Matter 10(21)*, 3826-2841. https://doi.org/10.1039/C3SM52959B

Berg, C.F., Slotte, P.A., Khanamiri, H.H. (2020). Geometrically derived efficiency of slow immiscible displacement in porous media. *Phys Rev E 102(3)*, https://doi.org/10.1103/PhysRevE.102.033113

Ibtehaz, N., Rahman, M.S. (2020). MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation. *Neural Networks 121*, 74-87. https://doi.org/10.1016/j.neunet.2019.08.025

Vikdal, Å.Å. (2021). *Segmentation of phases in images of fluid flow using deep learning*. Norges teknisk-naturvitenskaplige universitet.
