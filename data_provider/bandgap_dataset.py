"""
Band Gap Dataset for MOE Training

CSV Format:
- cif_string: CIF format structure
- robocrys_text: Robocrys description
- band_gap: Band gap value (eV)
- source_label: Data source (mp, jarvis_3d_optb88, etc.)
"""

import pandas as pd
import torch
import re
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph


class RobocrysProcessor:
    """Process Robocrys text: mask numbers with <num> token"""

    def __init__(self):
        # Regex pattern for numbers
        self.number_pattern = r'\b\d+\.?\d*\b'

    def mask_numbers(self, text):
        """
        Mask all numbers in text with <num> token

        Example:
            "The band gap is 2.5 eV" → "The band gap is <num> eV"
        """
        return re.sub(self.number_pattern, '<num>', text)

    def create_masked_template(self, source, robocrys_text):
        """
        Create masked template for band gap prediction

        Template: "{source}_bandgap is [MASK] eV. {robocrys}"

        Args:
            source: Source label (e.g., 'mp', 'jarvis_3d_optb88')
            robocrys_text: Robocrys description

        Returns:
            masked_text: Text with [MASK] token
        """
        # Mask numbers in robocrys
        masked_robocrys = self.mask_numbers(robocrys_text)

        # Create template
        template = f"{source}_bandgap is [MASK] eV. {masked_robocrys}"

        return template


class BandGapDataset(Dataset):
    """
    Dataset for Band Gap Prediction with MOE

    CSV columns:
        - cif_string: CIF format structure
        - robocrys_text: Robocrys description
        - band_gap: Band gap value (eV)
        - source_label: Data source
    """

    def __init__(
        self,
        csv_file,
        max_text_len=512,
        cutoff=8.0,
        max_neighbors=12,
    ):
        """
        Args:
            csv_file: Path to CSV file
            max_text_len: Maximum text length
            cutoff: Cutoff radius for graph construction
            max_neighbors: Maximum number of neighbors
        """
        self.df = pd.read_csv(csv_file)
        self.max_text_len = max_text_len
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        self.robocrys_processor = RobocrysProcessor()

        print(f"Loaded {len(self.df)} samples from {csv_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            g: Atom graph (DGL)
            lg: Line graph (DGL)
            text: Masked template text
            band_gap: Ground truth band gap
            source: Source label
        """
        row = self.df.iloc[idx]

        # 1. Load structure from CIF string
        cif_string = row['cif_string']
        atoms = Atoms.from_dict(eval(cif_string))  # If stored as dict string

        # Convert to ALIGNN graphs
        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            compute_line_graph=True,
            use_canonize=True,
        )

        # 2. Process text
        robocrys_text = row['robocrys_text']
        source = row['source_label']

        # Create masked template
        text = self.robocrys_processor.create_masked_template(source, robocrys_text)

        # 3. Band gap value
        band_gap = torch.tensor(row['band_gap'], dtype=torch.float32)

        return g, lg, text, band_gap, source


def collate_fn(batch):
    """
    Collate function for batching

    Args:
        batch: List of (g, lg, text, band_gap, source)

    Returns:
        Batched data
    """
    import dgl

    graphs, line_graphs, texts, band_gaps, sources = zip(*batch)

    # Batch graphs
    batched_graph = dgl.batch(graphs)
    batched_line_graph = dgl.batch(line_graphs)

    # Stack band gaps
    batched_band_gaps = torch.stack(band_gaps)

    return batched_graph, batched_line_graph, list(texts), batched_band_gaps, list(sources)


# Example CSV creation function
def create_example_csv():
    """Create example CSV file"""
    import pandas as pd

    data = {
        'cif_string': [
            "{'lattice': ..., 'coords': ..., 'elements': ...}",  # CIF dict
            "{'lattice': ..., 'coords': ..., 'elements': ...}",
        ],
        'robocrys_text': [
            "Silicon crystallizes in a cubic structure with space group Fd-3m.",
            "GaAs has a zinc blende structure with lattice parameter 5.65 angstrom.",
        ],
        'band_gap': [1.14, 1.42],
        'source_label': ['mp', 'jarvis_3d_optb88'],
    }

    df = pd.DataFrame(data)
    df.to_csv('data/example_bandgap.csv', index=False)
    print("Created example CSV: data/example_bandgap.csv")


if __name__ == '__main__':
    create_example_csv()
