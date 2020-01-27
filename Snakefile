"""
This is a re-factoring of the tractability calculations from:
    https://github.com/jurgjn/hotspots/blob/master/examples/1_tractibility/tractability.py#L109-L119

The aim is to parallelise the calculations using snakemake with the eventual aim of mapping
fragment hotspots proteome-wide using a combination of experimental and homology models.
"""

import os

import numpy as np

import pandas as pd

from ccdc.cavity import Cavity
from ccdc.protein import Protein
from ccdc.io import EntryWriter, MoleculeReader

from hotspots.calculation import Runner
from hotspots.hs_io import HotspotReader
from hotspots.hs_io import HotspotWriter
from hotspots.result import Extractor
from hotspots.pdb_python_api import PDBResult

subset = {
    '1e9x': 'd', '1udt': 'd', '2bxr': 'd', '1r9o': 'd', '3d4s': 'd', '1k8q': 'd',
    '1xm6': 'd', '1rwq': 'd', '1yvf': 'd', '2hiw': 'd', '1gwr': 'd', '2g24': 'd',
    '1c14': 'd', '1ywn': 'd', '1hvy': 'd', '1f9g': 'n', '1ai2': 'n', '2ivu': 'd',
    '2dq7': 'd', '1m2z': 'd', '2fb8': 'd', '1o5r': 'd', '2gh5': 'd', '1ke6': 'd',
    '1k7f': 'd', '1ucn': 'n', '1hw8': 'd', '2br1': 'd', '2i0e': 'd', '1js3': 'd',
    '1yqy': 'd', '1u4d': 'd', '1sqi': 'd', '2gsu': 'n', '1kvo': 'd', '1gpu': 'n',
    '1qpe': 'd', '1hvr': 'd', '1ig3': 'd', '1g7v': 'n', '1qmf': 'n', '1r58': 'd',
    '1v4s': 'd', '1fth': 'n', '1rsz': 'd', '1n2v': 'd', '1m17': 'd', '1kts': 'n',
    '1ywr': 'd', '2gyi': 'n', '1cg0': 'n', '5yas': 'n', '1icj': 'n', '1gkc': 'd',
    '1hqg': 'n', '1u30': 'd', '1nnc': 'n', '1c9y': 'n', '1j4i': 'd', '1qxo': 'n',
    '1o8b': 'n', '1nlj': 'n', '1rnt': 'n', '1d09': 'n', '1olq': 'n'
}

#subset = {'1j4i': 'd', '1rnt': 'n', } # Two small (~100 residue) structures for testing

rule all:
    """
        snakemake --cores 30 --dry-run
    """
    input:
        #expand('pdb/{pdb_id}.pdb', pdb_id = subset.keys()),
        #expand('pdb.prepare_protein/{pdb_id}.pdb', pdb_id = subset.keys()),
        #expand('pdb.prepare_protein.hotspots/{pdb_id}/out.zip', pdb_id = subset.keys()),
        expand('pdb.prepare_protein.hotspots.bcv/{pdb_id}/summary.tsv', pdb_id = subset.keys()),

rule pdb:
    """
    Download pdb structure, as in:
        https://github.com/prcurran/hotspots/blob/master/hotspots/calculation.py#L1159
    """
    output:
        pdb = 'pdb/{pdb_id}.pdb'
    run:
        PDBResult(identifier=wildcards.pdb_id).download(out_dir='pdb/')

rule prepare_protein:
    """
    Protein preparation and ligand removal, adopted from the CSD Python API docking example:
        https://downloads.ccdc.cam.ac.uk/documentation/API/cookbook_examples/docking_examples.html
    """
    input:
        pdb = '{prev_steps}/{pdb_id}.pdb'
    output:
        pdb = '{prev_steps}.prepare_protein/{pdb_id}.pdb'
    run:
        protein = Protein.from_file(input.pdb)
        protein.remove_all_waters()
        protein.remove_unknown_atoms()
        protein.add_hydrogens()
        ligands = protein.ligands
        for l in ligands:
            protein.remove_ligand(l.identifier)
        with EntryWriter(output.pdb) as writer:
            writer.write(protein)
        #print(f"{len(protein.residues)} residues found in {input.pdb} after prepare_protein")

rule hotspots:
    input:
        pdb = '{prev_steps}/{pdb_id}.pdb',
    output:
        zip = '{prev_steps}.hotspots/{pdb_id}/out.zip',
        pml = '{prev_steps}.hotspots/{pdb_id}/pymol_results_file.py',
    threads: 3 # Fragment hotspots maps is hard-coded to parallelise over three types of probes
    run:
        # 1) calculate Fragment Hotspot Result
        protein = Protein.from_file(input.pdb)
        r = Runner()
        result = r.from_protein(protein, buriedness_method='ghecom', nprocesses=3, probe_size=7)
        output_dir = os.path.dirname(output.zip)
        with HotspotWriter(output_dir) as w:
            w.write(result)

rule bcv:
    input:
        zip = '{prev_steps}/{pdb_id}/out.zip',
    output:
        zip = '{prev_steps}.bcv/{pdb_id}/out.zip',
        pml = '{prev_steps}.bcv/{pdb_id}/pymol_results_file.py',
        tsv = '{prev_steps}.bcv/{pdb_id}/summary.tsv',
    params:
        volume = 500,
        volume_default = 125, # https://github.com/prcurran/hotspots/blob/8944c24d0e23fea78debae76f04238d4a2618c31/build/lib/hotspots/result.py#L1027
        volume_failsafe = 5,
    run:
        # 1) Read Fragment Hotspot Result from disk
        # 2) calculate Best Continuous Volume
        try:
            result = HotspotReader(input.zip).read()
            extractor = Extractor(result)
            bcv_result = extractor.extract_volume(volume=params.volume)
            volume_out = params.volume

        except AssertionError:
            try:
                print(f"bcv with volume={params.volume} failed, re-running with volume={params.volume_default}")
                result_default = HotspotReader(input.zip).read()
                extractor_default = Extractor(result_default)
                bcv_result = extractor_default.extract_volume(volume=params.volume_default)
                volume_out = params.volume_default

            except AssertionError:
                print(f"bcv with volume={params.volume_default} failed, re-running with volume={params.volume_failsafe}")
                result_failsafe = HotspotReader(input.zip).read()
                extractor_failsafe = Extractor(result_failsafe)
                bcv_result = extractor_failsafe.extract_volume(volume=params.volume_failsafe)
                volume_out = params.volume_failsafe

        with HotspotWriter(os.path.dirname(output.zip)) as w:
            w.write(bcv_result)

        # 3) find the median score
        for probe, grid in bcv_result.super_grids.items():
            values = grid.grid_values(threshold=5)
            median = np.median(values)

        # 4) write tractability summary to .tsv
        results_table = pd.DataFrame({
            'scores': values,
            'pdb': [wildcards.pdb_id] * len(values),
            'median': [median] * len(values),
            'tractability': [subset[wildcards.pdb_id]] * len(values),
            'volume': [volume_out] * len(values)
        })
        results_table.to_csv(output.tsv, sep='\t', index=False)
