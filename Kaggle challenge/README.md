# Kernel methods Kaggle data challenge

Predicting whether a DNA sequence region is binding site to a specific transcription factor

- we should only use models that we have implemented our selves! no imports
- the task is binary classification

### Understanding the given data:
Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.


##### Protein sequences can be analysed by several tools

count_amino_acids: Simply counts the number times an amino acid is repeated in the protein sequence. Returns a dictionary {AminoAcid: Number} and also stores the dictionary in self.amino_acids_content.

get_amino_acids_percent: The same as count_amino_acids, only returns the number in percentage of entire sequence. Returns a dictionary and stores the dictionary in self.amino_acids_content_percent.

molecular_weight: Calculates the molecular weight of a protein.

aromaticity: Calculates the aromaticity value of a protein. It is simply the relative frequency of Phe+Trp+Tyr.

instability_index: This method tests a protein for stability. Any value above 40 means the protein is unstable (=has a short half life).

flexibility: Implementation of the flexibility method of Vihinen et al. (1994, Proteins, 19, 141-149).
isoelectric_point: This method uses the module IsoelectricPoint to calculate the pI of a protein.

secondary_structure_fraction: This methods returns a list of the fraction of amino acids which tend to be in helix, turn or sheet.
Amino acids in helix: V, I, Y, F, W, L.
Amino acids in turn: N, P, G, S.
Amino acids in sheet: E, M, A, L.
The list contains 3 values: [Helix, Turn, Sheet].

protein_scale(Scale, WindowSize, Edge): The method returns a list of values which can be plotted to view the change along a protein sequence.

reference link https://biopython.org/wiki/ProtParam
