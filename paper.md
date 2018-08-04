---
title: 'idpflex: Analysis of intrinsically disordered proteins by comparing MDsimulations to Small Angle Scattering experiments'
tags:
    - sans
    - neutron scattering
    - saxs
    - intrinsically disoreded proteins
    - simulations
    - MD
authors:
    - name: Jose M. Borreguero
      ORCID 0000-0002-0866-8158
      affiliation: 1
    - name: Fahima Islam
      ORCID 0000-0002-0265-0256
      affiliation: 1
    - name: Utsab R. Shrestha
      ORCID 0000-0001-6985-6657
      affiliation: 2
    - name: Loukas Petridis 
      ORCID 0000-0001-8569-060X
      affiliation: 2
affiliations:
 - name: Neutron Scattering Division, Oak Ridge National Laboratory, Oak Ridge TN, USA
   index: 1
 - name: Biosciences Division, Oak Ridge National Laboratory, Oak Ridge TN, USA.
   index: 2
date: 11 August 2018
bibliography: paper.bib
---

# Summary

It is estimated that about 30% of the proteome consists of Intrinsically disordered proteins
(IDP’s), yet their presence in public structural databases is severely
underrepresented. IDP’s adopt multiple conformations with similar probabilities,
preventing resolution of structures with X-Ray diffraction techniques. Small angle
scattering (SAS) probes the average features of the conformational ensemble, which can
prove unsatisfactory when very different ensembles share nearly identical average
features. Atomistic molecular dynamics (MD) simulations produce physically meaningful
conformations and offer a full-featured description of the conformational landscape
when properly validated against available SAS data. The python package idpflex
partitions the conformational ensemble resulting from a MD simulation into a
hierarchy of substates. Calculation of SAS intensities for each substate allows
quantitative comparison to SAS data, yielding the probability of the IDP to adopt the
conformation of a particular sub-state. idpflex can also compute other structural features
for the substates such as contact maps and secondary structure, and it’s extensible
to include other features on interest.

# Acknowledgements

We acknowledge contribution from .

# References
