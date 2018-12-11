---
title: 'idpflex: Analysis of Intrinsically Disordered Proteins by Comparing Simulations to Small Angle Scattering Experiments'
tags:
    - sans
    - neutron scattering
    - saxs
    - intrinsically disoreded proteins
    - simulations
    - molecular dynamics
authors:
    - name: Jose M. Borreguero
      orcid: 0000-0002-0866-8158
      affiliation: 1
    - name: Fahima Islam
      orcid: 0000-0002-0265-0256
      affiliation: 1
    - name: Utsab R. Shrestha
      orcid: 0000-0001-6985-6657
      affiliation: 2
    - name: Loukas Petridis 
      orcid: 0000-0001-8569-060X
      affiliation: 2
affiliations:
 - name: Neutron Scattering Division, Oak Ridge National Laboratory, Oak Ridge TN, USA
   index: 1
 - name: Biosciences Division, Oak Ridge National Laboratory, Oak Ridge TN, USA.
   index: 2
date: August 31 2018
bibliography: paper.bib
---

# Summary

It is estimated that about 30% of the eucariotic proteome consists of
intrinsically disordered proteins (IDP’s), yet their presence in public
structural databases is severely underrepresented.
IDP’s adopt heterogeneous inter-converting conformations with similar
probabilities, preventing resolution of structures with X-Ray
diffraction techniques. An alternative technique with wide application on
IDP systems is small angle scattering (SAS). SAS can measure average
structural features of IDP's when in vitro solution, or even at conditions
mimicking protein concentrations found in the cell's cytoplasm.

Despite these advantages, the averaging nature of SAS measurements
will prove unsatisfactory if one aims to differentiate among the different
conformations that a particular IDP can adopt. Different distributions
of conformations can yield the same average therefore it is not possible to
retrace the true distribution if all that SAS provides is the average
conformation.

To address this shortcoming, atomistic molecular dynamics (MD) simulations
of IDP systems combined with enhanced sampling methods such as the
Hamiltonian replica exchange method are specially
suitable [@Affentranger06]. These simulations can probe extensive
regions of the IDP's conformational space and have the potential to
offer a full-featured description of the conformational landscape of IDP's.
The results of these simulations should not be taken at faith value, however.
First, a proper comparison against available experimental SAS data
is a must. This validation step is the requirement that prompted the
development of `idpflex`.

The python package `idpflex` clusters the 3D conformations resulting from
an MD simulation into a hierarchical tree by means of structural similarity
among pairs of conformations. The conformations produced by the simulation
take the role of Leafs in the hierarchichal tree. Nodes in the tree take the
role of IDP substates, with conformations under a particular Node making up
one substate. Strictly speaking, `idfplex` does not require the
IDP conformations to be produced by an MD simulation. Alternative conformation
generators can be used, such as torsional sampling of the protein
backbone [@Curtis12].
In contrast to other methods [@Rozycki11], `idpflex` does not initially
discard any conformation by labelling it as incompatible with the
experimental data. This data is an average over
all conformations, and using this average as the criterion by which
to discard any specific conformation can lead to erroneous
discarding decisions due to the reasons stated above.

Default clustering is performed according to structural similarity
between pairs of conformations, defined by the root mean square deviation
algorithm [@Kabsch76]. Alternatively, `idpflex` can cluster
conformations according to an Euclidean distance in an abstract space
spanned by a set of structural properties, such as radius
of gyration and end-to-end distance. Comparison to experimental SAS data
is carried out first by calculating the SAS intensities [@Svergun95]
for each conformation produced by the MD simulation. This result in
SAS intensities for each Leaf in the hierarchical tree. Intensities are
then propagated up the hierarchical tree, yielding a SAS intensity for
each Node. Because each Node takes the role of a conformational substate,
we obtain SAS intensities for each substate. `idpflex` can compare
the SAS intensity of each substate against the experimental SAS data. Also,
it can average intensities from different substates and compare against
experimental SAS data. The fitting functionality included in `idpflex`
allows for selection of the set of substates that will yield
maximal similarity between computed and experimental SAS intensities. Thus,
arranging tens of thousands of conformations into (typically) less than
ten substates provides the researcher with a manageable number of
conformations from which to derive meaningful conclusions
regarding the conformational variability of IDP's.

`idpflex` also provides a set of convenience functions to compute
structural features of IDP's for each of the conformations produced by the MD
simulation. These properties can then be propagated up the hierarchical tree
much in the same way as SAS intensities are propagated. Thus, one can
compute for each substate properties such as radius of gyration,
end-to-end distance, asphericity, solvent exposed surface area,
contact maps, and secondary structure content. All these structural
properties require atomistic detail, thus `idpflex`
is more apt for the study of IDP's than for the study of quaternary protein
arrangements, where clustering of coarse-grain simulations becomes a
better option [@Rozycki11]. `idpflex` wraps other python
packages (MDAnalysis [@Michaud11], [@Gowers2016],
mdtraj [@McGibbon15])
and third party applications (CRYSOL [@Svergun95], DSSP [@Kabsch83])
that actually carry out the calculation of said properties.
Additional properties can be incorporated by inheriting from the
base Property classes.

To summarize, `idpflex` integrates MD simulations with SAS experiments in
order to obtain a manageable representation of the rich conformational
diversity of IDP's, a pertinent problem in structural biology.

The "notebooks" directory within the source contains two Jupyter
notebooks that illustrate the use of idpflex when clustering an
example MD trajectory.

# Notice of Copyright

This manuscript has been authored by UT-Battelle, LLC under Contract No.
DE-AC05-00OR22725 with the U.S. Department of Energy. The United States
Government retains and the publisher, by accepting the article for
publication, acknowledges that the United States Government retains a
non-exclusive, paid-up, irrevocable, worldwide license to publish or
reproduce the published form of this manuscript, or allow others to do
so, for United States Government purposes. The Department of Energy will
provide public access to these results of federally sponsored research
in accordance with the DOE Public Access Plan
(http://energy.gov/downloads/doe-public-access-plan).

# Acknowledgements

This work is sponsored by the Laboratory Directed Research and
Development Program of Oak Ridge National Laboratory, managed by
UT-Battelle LLC, for DOE. Part of this research is supported by the U.S.
Department of Energy, Office of Science, Office of Basic Energy
Sciences, User Facilities under contract number DE-AC05-00OR22725.

# References
